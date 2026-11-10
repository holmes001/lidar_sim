/**
 * @file    transport.h
 * @brief   Radar link: radar equation, delays, waveform generation, echo synthesis
 */
#ifndef LIDAR_TRANSPORT_H
#define LIDAR_TRANSPORT_H

#include <cstddef>

#include "typedef.h"

/**
 * @brief Radar parameters
 *
 * @note SI units throughout: range/height/wavelength m, area m^2, time s,
 *       frequency Hz, power W, amplitude sqrt(W) (consistent with sqrt(tx_p)).
 * @note Derived metrics: range resolution = c*dt/2; max unambiguous range =
 *       samp*dt*c/2.
 */
typedef struct
{
    std::size_t samp;                                ///< Number of samples
    double tx_p,tx_area,rx_area,wavelenth,dt,height; ///< TX power / TX-RX area / wavelength / sample interval / height
    double noise_sigma;                              ///< Noise stddev of each I/Q component
}radar;

/**
 * @brief Target parameters
 *
 * @note velocity is radial; its sign sets the Doppler direction (positive =
 *       approaching).
 */
typedef struct
{
    double area,distance,theta,velocity,height;   ///< Cross section / range / angle / radial velocity / height
}object;

/** @brief Waveform kind */
typedef enum { WF_PULSE, WF_SINE, WF_SQUARE, WF_IMPULSE, WF_FMCW } waveform_kind;

/**
 * @brief Waveform parameters, describing the complex envelope a(t), not the real
 *        received signal s(t)
 *
 * s(t) = Re{ a(t)*exp(j*2*pi*f0*t) }, f0 = c/lambda; |a| is the instantaneous
 * amplitude and arg(a) is the phase relative to the center carrier f0.
 *
 * @warning param is a union: only read and write the member matching kind.
 *          Assigning several members in a row overwrites them, and reading a
 *          non-active member is type punning with undefined results.
 * @warning Amplitude modulations (pulse, CW) have a real envelope; only a
 *          modulation whose instantaneous frequency deviates from f0 (FMCW) has
 *          a complex envelope.
 * @warning t_start_s shifts the matched filter peak as a whole: peak = echo delay
 *          + t_start_s, so it must be subtracted during ranging calibration.
 */
typedef struct {
    waveform_kind kind;   ///< Waveform kind

    union {
        struct { double width_s; double t_start_s; } pulse;    ///< Rectangular pulse: a = peak_amp
        struct { double period_s; }                  sine;     ///< Single-tone CW: a = peak_amp (constant)
        struct { double period_s; double duty; }     square;   ///< Square-wave window: a = +/-peak_amp
        struct { double period_s; double t_start_s; } impulse; ///< Impulse train: at most one non-zero sample per period
        /**
         * Linear chirp: a = peak_amp*exp(j*pi*S*(tau^2 - tau*T)), tau in [0, T).
         * S = bandwidth_hz / sweep_s, so the instantaneous baseband frequency
         * S*(tau - T/2) sweeps from -B/2 to +B/2; lambda is the center carrier
         * wavelength.
         */
        struct { double sweep_s; double bandwidth_hz; double t_start_s; } fmcw;
    } param;

    double peak_amp;   ///< Peak amplitude
} waveform;


/**
 * @brief Radar equation attenuation factor A = At*Ar*sigma/(4*pi*lambda^2*R^4)
 * @param tx_area    Transmit area [m^2]
 * @param rx_area    Receive area [m^2]
 * @param obj_area   Target cross section [m^2]
 * @param wavelength Wavelength [m]
 * @param distance   Range [m]
 * @return Power ratio; take sqrt to get the amplitude factor
 * @note Pure formula, numeric inputs only, easy to unit test and reuse.
 */
double decay_factor(double tx_area, double rx_area, double obj_area,
                    double wavelength, double distance);

/** @brief One-way delay R/c [s] */
double delay(double R);

/** @brief Round-trip delay 2R/c [s] */
double round_trip_delay(double R);

/**
 * @brief Generate the transmit complex envelope
 * @param[out] u          Output envelope, valid length
 *                        min(u.size(), samp * oversample)
 * @param      radar_1    Sampling context (samp / dt)
 * @param      wave_1     Waveform parameters
 * @param      oversample Record rate multiplier: u[j] sits at j*dt/oversample, and a
 *                        full receive window spans samp*oversample samples
 *
 * @note oversample > 1 buys delay accuracy with memory instead of arithmetic: the
 *       fractional part of the round-trip delay shrinks by the same factor, so a
 *       cheap linear kernel can reach the accuracy of a cubic one. In record mode
 *       recieve_signal must be told the same factor (echo_path::u_oversample).
 * @note The waveform is still evaluated in continuous time, so the oversampled
 *       record is a genuine sampling of the ideal envelope, not an interpolation
 *       of the coarse one.
 */
void generate_a_signal(ComplexVec &u,const radar &radar_1,waveform wave_1,
                       unsigned oversample = 1);

/**
 * @brief Extract one symbol from the transmit record as the matched filter reference
 * @param tx           Transmit record (output of generate_a_signal)
 * @param w            Waveform parameters, decide how to slice
 * @param r            Radar parameters, provides dt
 * @param u_oversample Record rate multiplier used when tx was generated
 * @return Reference template
 *
 * @note Finite envelopes (WF_PULSE / WF_FMCW) take the non-zero span, so the peak
 *       position maps directly to the echo delay; periodic envelopes (WF_SINE /
 *       WF_SQUARE / WF_IMPULSE) take one period.
 * @note u_oversample only matters for the periodic kinds, which slice by
 *       period/dt; passing the wrong factor yields a reference that is off by that
 *       factor. WF_PULSE / WF_FMCW detect their span from the data and are immune.
 * @warning param is a union: read the member matching kind. Reading a non-active
 *          member is type punning.
 */
ComplexVec make_reference(const ComplexVec &tx,const waveform &w,const radar &r,
                          unsigned u_oversample = 1);

/**
 * @brief Where the echo envelope comes from
 *
 * @note Record: the echo is a delayed copy of the transmit record u, so every
 *       imperfection the transmitter injects (phase noise, IQ imbalance, PA
 *       nonlinearity, DAC quantization, arbitrary measured waveforms) reaches the
 *       echo. This is the physical path and the default.
 * @note Analytic: the echo is re-evaluated from `wave` in continuous time. It has no
 *       interpolation error and does not read u at all, which makes it the reference
 *       implementation for unit tests, but it is structurally blind to any transmit
 *       imperfection. Keep it for cross-checks, not for system-level error studies.
 */
enum class TxSource { Record, Analytic };

/**
 * @brief Interpolation kernel used to shift the transmit record by a fractional sample
 *
 * @note Nearest: mandatory for WF_IMPULSE. A linear or cubic kernel would split one
 *       impulse sample into two half-amplitude samples and destroy "at most one
 *       non-zero sample per period"; the kind forces this kernel automatically.
 * @note Linear: cheapest, error grows with the squared normalized frequency; usually
 *       sufficient once u is oversampled (see echo_path::u_oversample).
 * @note Cubic: 4-point Lagrange, the default; smallest delay error for smooth
 *       envelopes (chirps, CW) at 4 multiply-adds per sample.
 */
enum class DelayKernel { Nearest, Linear, Cubic };

/**
 * @brief How the echo is synthesized
 *
 * @note u_oversample states the transmit record rate: u[j] sits at j*dt/u_oversample.
 *       With u_oversample > 1 the fractional part of the delay shrinks by the same
 *       factor, so a Linear kernel can match Cubic quality -- accuracy bought with
 *       memory instead of arithmetic.
 * @note u_oversample is ignored by TxSource::Analytic and must match the factor that
 *       generate_a_signal used to build u.
 */
struct echo_path
{
    TxSource    source       = TxSource::Record;   ///< Envelope origin
    DelayKernel kernel       = DelayKernel::Cubic; ///< Fractional-delay kernel (Record only)
    unsigned    u_oversample = 1;                  ///< u sampling rate = dt/u_oversample
};

/**
 * @brief Synthesize the echo of a single target
 * @param u        Transmit record from generate_a_signal; sampled at
 *                 dt/path.u_oversample
 * @param radar_1  Radar parameters (including noise_sigma)
 * @param object_1 Target parameters
 * @param wave     Waveform parameters; read only by TxSource::Analytic
 * @param t_offset Absolute time offset of the record start [s]; pass p*PRI for
 *                 pulse p
 * @param path     Envelope source, delay kernel and transmit record rate
 * @return Echo envelope of length samp
 *
 * @note The delay is continuous in both modes: record mode resamples u at the
 *       fractional lag with path.kernel, analytic mode samples `wave` at that same
 *       lag. Neither quantizes the delay to whole samples.
 * @note Record mode falls back to the analytic evaluation when u is empty, so legacy
 *       callers that pass an empty record still get a meaningful echo.
 * @note Noise is receiver-side and depends only on noise_sigma, independent of range
 *       or target presence.
 * @warning WF_IMPULSE ignores path.kernel and always resamples nearest-neighbour.
 * @warning t_offset determines whether the Doppler phase accumulates across pulses:
 *          always passing 0 restarts the Doppler phase at every pulse, making the
 *          slow-time sequence constant and the Doppler always zero.
 * @warning In record mode the record must cover the largest lag you expect. Lags
 *          shifted past its end are truncated to zero, which the analytic path never
 *          does -- the two modes disagree there by design.
 */
ComplexVec recieve_signal(
    const ComplexVec &u,const radar &radar_1,const object &object_1,
    const waveform &wave,double t_offset = 0.0,
    const echo_path &path = echo_path());

/**
 * @brief Acquisition parameters of one pulse-Doppler frame
 *
 * @warning pri must be >= samp*dt: if the receive window is longer than one PRI, the
 *          window of pulse p covers the echo of pulse p+1 (pulse crosstalk).
 * @note Unambiguous range = c*PRI/2; Doppler resolution = 1/(n_pulse*PRI).
 */
typedef struct
{
    int    n_pulse;   ///< Pulses per frame N (i.e. slow-time samples)
    double pri;       ///< Pulse repetition interval [s]
}pulse_doppler_frame;

/**
 * @brief Window each pulse per the transmit schedule and assemble one frame
 * @param u        Transmit record, forwarded to recieve_signal (so transmitter
 *                 errors in u reach every pulse)
 * @param radar_1  Radar parameters; radar_1.samp is the receive window per pulse
 * @param object_1 Target parameters
 * @param wave     Waveform parameters (read only by TxSource::Analytic)
 * @param frame    Frame parameters (pulse count, PRI)
 * @param path     Envelope source, delay kernel and transmit record rate
 * @return Echo matrix, rows = pulse index (slow time), cols = fast-time samples,
 *         size n_pulse x samp
 * @throws std::invalid_argument n_pulse <= 0, pri <= 0, or samp*dt > pri
 *
 * @note Row index p follows the **transmit schedule** t = p*PRI and never relies on
 *       peak detection: the slow-time phase reference must be anchored to the
 *       transmit instants; using detected peaks to assign rows causes phase breaks
 *       and Doppler spectral spreading when a pulse is missed.
 * @note In record mode the record length must cover samp*dt + the largest lag; the
 *       per-pulse window is still samp samples.
 */
ComplexVec2D recieve_frame(
    const ComplexVec &u,const radar &radar_1,const object &object_1,
    const waveform &wave,const pulse_doppler_frame &frame,
    const echo_path &path = echo_path());

#endif
