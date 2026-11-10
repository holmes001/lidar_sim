/**
 * @file    transport.cpp
 * @brief   Radar link implementation: radar equation, delays, envelope waveforms,
 *          echo synthesis
 */
#include "transport.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <complex>

#include "random_error.h"

/**
 * @brief Noise figure (deprecated)
 * @deprecated Noise is now driven by radar::noise_sigma; this variable takes part in
 *             no computation and is kept only for compatibility with legacy references.
 */
double noise_factor = 0.1;

/**
 * @brief Global random seed
 * @note Incremented on every recieve_signal call: echoes within one process get
 *       uncorrelated noise, while rerunning the same code reproduces the result.
 */
unsigned int g_seed = 8051;

/// Radar equation attenuation factor (numeric inputs only, easy to test and reuse)
double decay_factor(double tx_area, double rx_area, double obj_area,
                    double wavelength, double distance)
{
    const double R2 = distance * distance;
    const double R4 = R2 * R2;
    const double lambda2 = wavelength * wavelength;
    return (rx_area * tx_area * obj_area) / (4.0 * kpi * lambda2 * R4);
}

/// One-way delay R/c [s]
double delay(double R)
{
    return R / c;
}

/// Round-trip delay 2R/c [s]
double round_trip_delay(double R)
{
    return 2.0 * R / c;
}

/**
 * @brief Complex envelope at a continuous time t
 * @param w  Waveform parameters
 * @param t  Time instant [s]
 * @param dt Sampling interval; used by the impulse train only
 * @return Envelope value; 0 outside the window
 *
 * @note Amplitude modulations (pulse, CW) return a real envelope; only FMCW, whose
 *       instantaneous frequency deviates from f0, returns a complex one.
 * @note The pulse window is half-open, consistently at the transmitter and the echo,
 *       so a boundary landing exactly on a sample does not cost one sample off.
 */
static Complex wf_eval(const waveform &w, double t, double dt)
{
    const Complex zero(0.0, 0.0);

    switch (w.kind)
    {
    case WF_PULSE:
    {
        const double t_start = w.param.pulse.t_start_s;
        const double tau     = w.param.pulse.width_s;
        return (t >= t_start && t < t_start + tau) ? Complex(w.peak_amp, 0.0) : zero;
    }
    case WF_SINE:
        /// Single-tone CW: s(t) = peak_amp*cos(2*pi*f0*t), so the envelope is constant
        return Complex(w.peak_amp, 0.0);

    case WF_SQUARE:
    {
        const double T = w.param.square.period_s;
        const double d = w.param.square.duty;
        if (T <= 0.0 || d <= 0.0) return zero;
        double u = std::fmod(t, T) / T;
        if (u < 0.0) u += 1.0;
        return Complex((u < d) ? w.peak_amp : -w.peak_amp, 0.0);
    }
    case WF_IMPULSE:
    {
        const double t_start = w.param.impulse.t_start_s;
        const double period  = w.param.impulse.period_s;
        if (period <= 0.0 || dt <= 0.0) return zero;

        const double d    = t - t_start;
        const double half = 0.5 * dt;
        const double eps  = 1.0e-6 * dt;   ///< Absorbs float rounding, avoids missing an impulse at the half-sample boundary

        /// Only an offset to the nearest impulse instant within [-dt/2, dt/2) counts as a hit;
        /// the half-open interval yields exactly one non-zero sample even when the instant
        /// falls midway between two samples. The lower bound carries a tolerance: with a
        /// fractional delay, the sample nearest the first impulse may lie slightly before
        /// t_start and must still count.
        const double k   = std::floor(d / period + 0.5);
        const double off = d - k * period;
        return (off >= -half - eps && off < half - eps) ? Complex(w.peak_amp, 0.0) : zero;
    }
    case WF_FMCW:
    {
        const double T   = w.param.fmcw.sweep_s;
        const double B   = w.param.fmcw.bandwidth_hz;
        const double tau = t - w.param.fmcw.t_start_s;
        if (T <= 0.0 || tau < 0.0 || tau >= T) return zero;

        const double S = B / T;                            ///< Chirp slope [Hz/s]
        /// phase = pi*S*(tau^2 - tau*T), instantaneous baseband frequency = S*(tau - T/2) in [-B/2, +B/2]
        const double phase = kpi * S * (tau * tau - tau * T);
        return Complex(w.peak_amp * std::cos(phase), w.peak_amp * std::sin(phase));
    }
    }
    return zero;
}

/**
 * @brief Nearest-neighbour resample of the transmit record at a fractional index
 *
 * Required for impulse trains: a linear or cubic kernel would split one impulse
 * sample into two half-amplitude samples, breaking "at most one non-zero sample per
 * period" and smearing the pulse energy over an extra sample.
 */
static Complex pick_nearest(const ComplexVec &u, double pos)
{
    if (u.empty()) return Complex(0.0, 0.0);
    const long i = std::lround(pos);
    if (i < 0 || static_cast<std::size_t>(i) >= u.size()) return Complex(0.0, 0.0);
    return u[static_cast<std::size_t>(i)];
}

/// Linear (2-point) resample: cheapest kernel, error grows with the squared normalized frequency
static Complex pick_linear(const ComplexVec &u, double pos)
{
    if (u.empty() || pos < 0.0 || pos > static_cast<double>(u.size() - 1))
        return Complex(0.0, 0.0);

    const std::size_t i0 = static_cast<std::size_t>(pos);
    const double f = pos - static_cast<double>(i0);
    if (i0 + 1 >= u.size()) return u[i0];
    return u[i0] * (1.0 - f) + u[i0 + 1] * f;
}

/**
 * @brief 4-point Lagrange (cubic) resample at a fractional index
 * @note Samples outside the record are taken as zero: the transmitter has stopped, so
 *       the record edge is a genuine truncation, not a periodic wrap.
 */
static Complex pick_cubic(const ComplexVec &u, double pos)
{
    if (u.empty() || pos < 0.0 || pos > static_cast<double>(u.size() - 1))
        return Complex(0.0, 0.0);

    const long i1 = static_cast<long>(pos);
    const double f = pos - static_cast<double>(i1);
    const auto at = [&u](long k) -> Complex {
        if (k < 0 || static_cast<std::size_t>(k) >= u.size()) return Complex(0.0, 0.0);
        return u[static_cast<std::size_t>(k)];
    };

    /// Lagrange weights over the nodes {-1, 0, +1, +2}; w(-1)+w(0)+w(1)+w(2) = 1 for any f
    const double wm = -f * (f - 1.0) * (f - 2.0) / 6.0;
    const double w0 =  (f + 1.0) * (f - 1.0) * (f - 2.0) / 2.0;
    const double w1 = -(f + 1.0) * f * (f - 2.0) / 2.0;
    const double w2 =  (f + 1.0) * f * (f - 1.0) / 6.0;

    return at(i1 - 1) * wm + at(i1) * w0 + at(i1 + 1) * w1 + at(i1 + 2) * w2;
}

/// Fractional-delay read of the transmit record at fine-sample index pos
static Complex interp_record(const ComplexVec &u, double pos, DelayKernel kernel)
{
    switch (kernel)
    {
    case DelayKernel::Nearest: return pick_nearest(u, pos);
    case DelayKernel::Linear:  return pick_linear(u, pos);
    case DelayKernel::Cubic:   return pick_cubic(u, pos);
    }
    return Complex(0.0, 0.0);
}

void generate_a_signal(ComplexVec &u,const radar &radar_1,waveform wave_1,
                       unsigned oversample)
{
    const unsigned os = (oversample == 0u) ? 1u : oversample;
    const double dt_fine = radar_1.dt / static_cast<double>(os);
    /// u is sampled at dt/os, so one receive window spans samp*os samples
    const std::size_t n = std::min(u.size(), radar_1.samp * static_cast<std::size_t>(os));
    for(std::size_t i=0;i<n;i++)
    {
        /// Evaluate in continuous time throughout: this automatically turns the impulse
        /// train into "at most one non-zero sample per period", at any record rate
        u[i] = wf_eval(wave_1, static_cast<double>(i) * dt_fine, dt_fine);
    }
}

/**
 * @brief Extract one symbol from the transmit record as the matched filter reference
 * @param tx Transmit complex envelope
 * @param w  Waveform parameters, decide how to slice
 * @param r  Radar parameters, provides dt
 * @return Reference template
 */
ComplexVec make_reference(const ComplexVec &tx,
                          const waveform &w, const radar &r, unsigned u_oversample)
{
    ComplexVec ref;
    if (tx.empty()) return ref;

    if (w.kind == WF_PULSE || w.kind == WF_FMCW)
    {
        std::size_t lo = 0, hi = tx.size();
        while (lo < hi && std::abs(tx[lo]) == 0.0) ++lo;
        while (hi > lo && std::abs(tx[hi - 1]) == 0.0) --hi;
        if (lo < hi)
            ref.assign(tx.begin() + static_cast<std::ptrdiff_t>(lo),
                       tx.begin() + static_cast<std::ptrdiff_t>(hi));
        return ref;
    }

    double period = 0.0;
    switch (w.kind)
    {
    case WF_SINE:    period = w.param.sine.period_s;    break;
    case WF_SQUARE:  period = w.param.square.period_s;  break;
    case WF_IMPULSE: period = w.param.impulse.period_s; break;
    default:         period = 0.0;                      break;
    }

    std::size_t len = (period > 0.0 && r.dt > 0.0)
                    ? static_cast<std::size_t>(std::llround(
                          period / (r.dt / static_cast<double>((u_oversample == 0u) ? 1u : u_oversample))))
                    : 0;
    const std::size_t cap = std::max<std::size_t>(2, tx.size() / 8);
    len = std::min(std::max<std::size_t>(len, 2), cap);
    len = std::min(len, tx.size());

    ref.assign(tx.begin(), tx.begin() + len);
    return ref;
}


ComplexVec recieve_signal(
    const ComplexVec &u,const radar &radar_1,const object &object_1,
    const waveform &wave,double t_offset,const echo_path &path)
{
    ComplexVec v(radar_1.samp);

    /// The radar equation gives a power ratio; take the square root for the amplitude factor
    const double A = std::sqrt(
        decay_factor(radar_1.tx_area, radar_1.rx_area, object_1.area,
                     radar_1.wavelenth, object_1.distance));

    /// The round-trip delay is continuous, so evaluate directly at t = t_offset + (i - tau/dt)*dt.
    /// Avoid writing i*dt - tau: with both terms near 2e-5 s the subtraction loses two
    /// significant digits and disagrees with the transmit-side t = i*dt test at the boundary.
    /// t_offset lets the Doppler phase accumulate across pulses (pass p*PRI for pulse p).
    const double tau    = round_trip_delay(object_1.distance);
    const double n_tau  = tau / radar_1.dt;          ///< Fractional-sample delay, in dt units
    const double fd     = 2.0 * object_1.velocity / radar_1.wavelenth;   ///< Doppler [Hz]

    /// Round-trip carrier phase phi0 = k*2R = 4*pi*R/lambda (modulo 2*pi). Doppler is only its
    /// time derivative, so the constant term must be kept separately: coherent integration,
    /// interferometric ranging and multi-target phase relations all need it.
    const double phi0 = std::fmod(2.0 * kpi * (2.0 * object_1.distance / radar_1.wavelenth),
                                  2.0 * kpi);

    /// Record mode: the echo is a delayed copy of what the transmitter actually radiated, so
    /// transmitter errors already present in u reach the echo. An empty record falls back to the
    /// analytic evaluation to keep legacy callers working.
    const bool   use_record = (path.source == TxSource::Record) && !u.empty();
    const double u_os = static_cast<double>((path.u_oversample == 0u) ? 1u : path.u_oversample);
    /// An impulse train must keep at most one non-zero sample per period, so a kernel that
    /// spreads energy over two samples is forbidden regardless of what the caller asked for
    const DelayKernel kernel = (wave.kind == WF_IMPULSE) ? DelayKernel::Nearest : path.kernel;

    for(std::size_t i = 0; i < radar_1.samp; i++)
    {
        /// The envelope is evaluated at the lag relative to this pulse's transmit instant, so it
        /// is independent of t_offset: mixing t_offset in would push t outside the waveform
        /// window from the second pulse on, making the echo identically zero.
        const double lag     = static_cast<double>(i) - n_tau;
        const double t_local = lag * radar_1.dt;
        /// The carrier phase is evaluated at **absolute** time: t_offset is exactly what makes the
        /// Doppler phase accumulate across pulses.
        const double t_abs   = t_offset + t_local;

        /// lag is measured in dt; the transmit record is sampled at dt/u_os, so index it in
        /// fine samples
        const Complex env = use_record
                          ? interp_record(u, lag * u_os, kernel)
                          : wf_eval(wave, t_local, radar_1.dt);

        v[i] = A * env * std::exp(Complex(0.0, 2.0 * kpi * fd * t_abs + phi0));
    }

    /// Noise is a receiver property, independent of range and of target presence; incrementing
    /// g_seed on every call keeps successive echoes uncorrelated
    return addGaussianNoiseStd(v, radar_1.noise_sigma, 0.0, g_seed++);
}


ComplexVec2D recieve_frame(
    const ComplexVec &u,const radar &radar_1,const object &object_1,
    const waveform &wave,const pulse_doppler_frame &frame,const echo_path &path)
{
    if (frame.n_pulse <= 0)
        throw std::invalid_argument("n_pulse must be > 0");
    if (frame.pri <= 0.0)
        throw std::invalid_argument("pri must be > 0");

    /// The receive window must not span a PRI, otherwise the window of pulse p covers the
    /// echo of pulse p+1
    const double window = static_cast<double>(radar_1.samp) * radar_1.dt;
    if (window > frame.pri)
        throw std::invalid_argument("receive window (samp*dt) must not exceed pri");

    /// Row indices follow the transmit schedule t = p*PRI with no peak detection:
    /// the slow-time phase reference must be anchored to the transmit instants.
    ComplexVec2D echoes;
    echoes.reserve(static_cast<std::size_t>(frame.n_pulse));
    for (int p = 0; p < frame.n_pulse; ++p)
    {
        echoes.push_back(recieve_signal(u, radar_1, object_1, wave,
                                        static_cast<double>(p) * frame.pri, path));
    }
    return echoes;
}
