/**
 * @file    main.cpp
 * @brief   Demo: transmit -> echo -> closed-loop power amplification -> matched
 *          filter ranging
 *
 * @details Flow: generate_a_signal builds the complex envelope -> recieve_signal
 *          synthesizes a noisy echo -> amplifyToTargetPower drives it to the target
 *          power -> matchedFilter performs pulse compression -> the peak position is
 *          converted to a range. Each intermediate result is saved under figs/.
 */
#include "transport.h"      ///< radar / object / waveform, waveform generation, echo
#include "filter.h"         ///< Matched filtering
#include "doppler.h"        ///< Range-Doppler
#include "power.h"          ///< Power measurement and closed-loop amplification
#include "app.h"            ///< FFT / sampling / axes / figure saving
#include "plot.h"           ///< Plotting (pl::Style / cv::Mat)

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

/// Figure file names (relative to the figs/ directory); save_picture joins the full path
const std::string fig_pulse = "pulse_with_error.png";
const std::string fig_pid   = "pid.png";
const std::string fig_mf    = "matched_filter.png";
const std::string fig_tx    = "tx_waveform.png";
const std::string fig_rd    = "range_doppler.png";
const std::string fig_rd_r  = "rd_range_cut.png";
const std::string fig_rd_v  = "rd_doppler_cut.png";

int main()
{
    //parameters

    radar radar_1;
    radar_1.rx_area = 1;                 ///< Receive area [m^2]
    radar_1.tx_area = 1;                 ///< Transmit area [m^2]
    radar_1.samp = 24000;                ///< Sample count; max unambiguous range = samp*dt*c/2 = 3600 m
    radar_1.dt = 1e-9;                   ///< Sample interval; range resolution c*dt/2 = 0.15 m
    radar_1.wavelenth = 0.03;            ///< Center wavelength [m]
    radar_1.tx_p = 1.0;                  ///< Transmit power [W]
    radar_1.height = 1000;               ///< Height [m]
    radar_1.noise_sigma = 2e-8;          ///< Noise stddev of each I/Q component

    waveform wave_1;
    wave_1.kind = WF_PULSE;
    wave_1.peak_amp = sqrt(radar_1.tx_p);

    wave_1.param.pulse.width_s = 100*radar_1.dt;   ///< Pulse width of 100 sample (100 ns)
    wave_1.param.pulse.t_start_s = 0.0;   ///< Pulse start at t = 0

    /*
    wave_1.param.impulse.period_s = 1e-6;   ///< Pulse repetition interval 1 us
    wave_1.param.impulse.t_start_s = 0.0;   ///< At most one non-zero sample per period, starting at 0
    */
    /*
    wave_1.param.fmcw.t_start_s    = 0.0;
    wave_1.param.fmcw.sweep_s      = 1e-6;    ///< Sweep duration 1 us
    wave_1.param.fmcw.bandwidth_hz = 100e6;   ///< Bandwidth 100 MHz -> range resolution c/2B = 1.5 m
    */

    object object_1;
    object_1.area = 0.2;
    object_1.distance = 3000;
    object_1.theta = kpi;
    object_1.velocity = 20;

    ComplexVec a(radar_1.samp);
    ComplexVec rs_1(radar_1.samp);
    ComplexVec rs_2(radar_1.samp);
    ComplexVec rs_3(radar_1.samp);


    generate_a_signal(a,radar_1,wave_1);

    rs_1 = recieve_signal(a,radar_1,object_1,wave_1);

    std::vector<double> hist;

    PowerLoopConfig cfg;
    cfg.kp      = 0.8;
    cfg.ki      = 0.1;
    cfg.tol_rel = 1e-4;
    cfg.history = &hist;

    PowerLoopReport rep = amplifyToTargetPower(rs_1, 1e-3, cfg);
    ComplexVec amplified  = rep.output ;

    /// Matched filter reference: slice one symbol according to the waveform kind
    ComplexVec ref = make_reference(a, wave_1, radar_1);

    /// Same mode: output length equals input length, and index t is "matching starting
    /// at sample t", so the peak index is exactly the sample where the echo arrives
    ComplexVec mf = matchedFilter(amplified, ref, MatchFilterMode::Same, false);

    std::vector<double> mf_mag = vector_abs(mf);

    /// Real part of the baseband envelope (output of generate_a_signal) and of the
    /// amplified echo
    std::vector<double> tx_real = vector_real(a);
    std::vector<double> rx_real = vector_real(amplified);

    /// Time-domain plots use time [us] on x; the matched filter plot uses range [m]
    const std::vector<double> t_us = time_axis(radar_1.samp, radar_1.dt);
    const std::vector<double> r_m  = rangeAxis(radar_1.samp, radar_1.dt);

    /// Crop the time-domain plots: the transmit window is only 1 us while the echo sits
    /// near 20 us, so the full 24 us record is unreadable
    std::vector<double> t_tx, y_tx, t_rx, y_rx;
    crop_window(t_us, tx_real, 0.0, 2.0, t_tx, y_tx);
    crop_window(t_us, rx_real, 19.0, 22.0, t_rx, y_rx);

    pl::Style st_tx;
    st_tx.title  = "Baseband envelope Re{a(t)} (FMCW, 0-2 us)";
    st_tx.xlabel = "Time (us)";
    st_tx.ylabel = "Amplitude";
    st_tx.legend = {"I(t)"};
    save_figure(pl::PlotLineXY(pl::Make1D(t_tx), pl::Make1D(y_tx), st_tx), fig_tx);

    pl::Style st;
    st.title  = "Received echo after power loop (19-22 us)";
    st.xlabel = "Time (us)";
    st.ylabel = "Amplitude";
    st.legend = {"rx (I)"};
    save_figure(pl::PlotLineXY(pl::Make1D(t_rx), pl::Make1D(y_rx), st), fig_pulse);

    /// Power loop convergence curve: x is the iteration index 0..N-1, so PlotLine suffices
    pl::Style st_1;
    st_1.title  = "Power loop convergence";
    st_1.xlabel = "iteration";
    st_1.ylabel = "Power (W)";
    st_1.legend = {"P_meas"};
    save_picture(pl::Make1D(hist), st_1, fig_pid);

    pl::Style st_2;
    st_2.title  = "Matched filter output";
    st_2.xlabel = "Range (m)";
    st_2.ylabel = "|y(tau)|";
    st_2.legend = {"MF"};
    save_figure(pl::PlotLineXY(pl::Make1D(r_m), pl::Make1D(mf_mag), st_2), fig_mf);

    /// Peak position -> target range (range = sample index * c*dt/2)
    const std::size_t pk = peakPowerIndex(mf);
    std::cout << "matched filter peak at sample " << pk
              << "  =>  range = " << pk * radar_1.dt * c / 2.0 << " m\n"
              << "peak/mean power = "
              << powerToDB(peakPower(mf), measureSignalPower(mf)) << " dB"
              << std::endl;

    /// ---- range-Doppler: pulse train + slow-time FFT ----
    /// Frame parameters are defined together; row indices come from the transmit
    /// schedule t = p*PRI, with no peak detection.
    pulse_doppler_frame frame;
    frame.n_pulse = 64;        ///< Pulses per frame N; Doppler resolution = 1/(N*PRI) = 156 Hz
    frame.pri     = 100e-6;    ///< Pulse repetition interval [s]; unambiguous range = c*PRI/2 = 15 km

    ComplexVec2D pulses = recieve_frame(a, radar_1, object_1, wave_1, frame);

    /// Aliases used by the plotting and printing code below
    const int    n_pulse = frame.n_pulse;
    const double pri     = frame.pri;

    /// Fast time: matched filter per pulse -> slow-time matrix slow[p][r]
    ComplexVec2D slow = matchedFilterFrame(pulses, ref, MatchFilterMode::Same);

    /// Slow time: N-point FFT per range gate -> range-Doppler map rd[k][r]
    ComplexVec2D rd = rangeDopplerMap(slow);

    const RangeDopplerPeak peak_rd =
        findRangeDopplerPeak(rd, pri, radar_1.dt, radar_1.wavelenth);

    const double fd_theory = 2.0 * object_1.velocity / radar_1.wavelenth;
    std::cout << "range-doppler: N = " << n_pulse
              << ", PRI = " << pri * 1e6 << " us"
              << ", doppler resolution = " << 1.0 / (n_pulse * pri) << " Hz\n"
              << "  range    = " << peak_rd.range_m
              << " m   (theory " << object_1.distance << ")\n"
              << "  doppler  = " << peak_rd.doppler_hz
              << " Hz  (theory " << fd_theory << ")\n"
              << "  velocity = " << peak_rd.velocity
              << " m/s (theory " << object_1.velocity << ")\n";

    /// Phase difference of adjacent pulses at the peak range gate; should equal 2*pi*fd*PRI
    const std::size_t gate = peak_rd.range_index;
    if (slow.size() > 1 && gate < slow[1].size())
    {
        const double dphi = std::arg(slow[1][gate] * std::conj(slow[0][gate]));
        std::cout << "  phase step between pulses = " << dphi << " rad"
                  << "   (theory 2*pi*fd*PRI = " << 2.0 * kpi * fd_theory * pri << " rad)\n";
    }

    /// Color the range-Doppler map in dB relative to the global peak: on a linear scale the
    /// sinc sidelobes and the noise floor collapse into solid black, leaving one bright dot.
    const double ref_amp = std::sqrt(peak_rd.power);   ///< Global peak amplitude
    const auto to_db = [ref_amp](const Complex &v) {
        const double a = std::abs(v) / ref_amp;
        return 20.0 * std::log10(std::max(a, 1e-6));   ///< Floor at -120 dB to avoid log(0)
    };

    if (!rd.empty() && !rd[0].empty())
    {
        const std::size_t n_range = rd[0].size();
        const std::vector<double> range_all = rangeAxis(n_range, radar_1.dt);

        /// Heat map: keep only a window of range gates around the peak, otherwise the
        /// 24000 columns collapse into a single line
        const std::size_t half_win = 120;
        const std::size_t r0 = (peak_rd.range_index > half_win)
                             ? peak_rd.range_index - half_win : 0;
        const std::size_t r1 = std::min(n_range, peak_rd.range_index + half_win + 1);

        std::vector<std::vector<double>> rd_db(n_pulse, std::vector<double>(r1 - r0));
        for (int k = 0; k < n_pulse; ++k)
            for (std::size_t r = r0; r < r1; ++r)
                rd_db[k][r - r0] = to_db(rd[k][r]);

        pl::Style st_rd;
        /// Note: the y label of a pseudo-color map is drawn horizontally centered at
        /// margin_left (not rotated), so a long label would overflow and be clipped;
        /// keep it as "doppler bin" and put the unit in the title.
        st_rd.title    = "Range-Doppler (dB re peak, 156 Hz/bin)";
        st_rd.xlabel   = "range bin (0.15 m each)";
        st_rd.ylabel   = "doppler bin";
        st_rd.auto_y   = false;   ///< In a pseudo-color map y_min/y_max act as the color scale vmin/vmax
        st_rd.y_min    = -60.0;
        st_rd.y_max    = 0.0;
        st_rd.decimals = 0;
        save_figure(pl::PlotPseudoColor(pl::Make2D(rd_db), st_rd,
                                        cv::COLORMAP_TURBO, true), fig_rd);

        /// Range profile: fix the peak Doppler gate and use physical range on x.
        /// Keep the span moderate: aliasing ripples appear unless the 0.15 m/sample rate
        /// stays close to the pixel sampling ratio.
        const std::size_t cut_half = 400;
        const std::size_t c0 = (peak_rd.range_index > cut_half)
                             ? peak_rd.range_index - cut_half : 0;
        const std::size_t c1 = std::min(n_range, peak_rd.range_index + cut_half + 1);

        std::vector<double> cut_r(c1 - c0);
        for (std::size_t r = c0; r < c1; ++r)
            cut_r[r - c0] = to_db(rd[peak_rd.doppler_index][r]);
        const std::vector<double> cut_r_axis(range_all.begin() + static_cast<std::ptrdiff_t>(c0),
                                             range_all.begin() + static_cast<std::ptrdiff_t>(c1));

        pl::Style st_cut_r;
        st_cut_r.title    = "Range profile at peak Doppler";
        st_cut_r.xlabel   = "Range (m)";
        st_cut_r.ylabel   = "|RD| (dB re peak)";
        st_cut_r.legend   = {"range cut"};
        st_cut_r.decimals = 0;
        save_figure(pl::PlotLineXY(pl::Make1D(cut_r_axis), pl::Make1D(cut_r), st_cut_r),
                    fig_rd_r);

        /// Doppler spectrum: fix the peak range gate and use radial velocity on x
        const std::vector<double> v_axis =
            velocityAxis(n_pulse, pri, radar_1.wavelenth);
        std::vector<double> cut_v(n_pulse);
        for (int k = 0; k < n_pulse; ++k)
            cut_v[k] = to_db(rd[k][peak_rd.range_index]);

        pl::Style st_cut_v;
        st_cut_v.title        = "Doppler spectrum at peak range";
        st_cut_v.xlabel       = "Velocity (m/s)";
        st_cut_v.ylabel       = "|RD| (dB re peak)";
        st_cut_v.legend       = {"doppler cut"};
        st_cut_v.decimals     = 0;
        st_cut_v.show_markers = true;   ///< Only N points, so markers make it easier to read
        save_figure(pl::PlotLineXY(pl::Make1D(v_axis), pl::Make1D(cut_v), st_cut_v),
                    fig_rd_v);
    }

    /// ---- cross-check: analytic vs record path under a fractional-sample delay ----
    /// Runs last on purpose: recieve_signal advances g_seed, so calling it earlier would
    /// shift the noise realisation of every figure above.
    {
        radar quiet = radar_1;
        quiet.noise_sigma = 0.0;          ///< Noise would mask the difference between the paths

        object frac_target = object_1;
        frac_target.distance = 3000.375;  ///< 2R/(c*dt) = 20000.25 -> forces a fractional shift
        frac_target.velocity = 0.0;       ///< Drop Doppler so only the envelope synthesis differs

        const ComplexVec ref_chk = make_reference(a, wave_1, radar_1);

        echo_path p_analytic;
        p_analytic.source = TxSource::Analytic;

        echo_path p_linear;
        p_linear.source = TxSource::Record;
        p_linear.kernel = DelayKernel::Linear;

        echo_path p_cubic;                ///< Record mode (default) + cubic kernel
        p_cubic.source = TxSource::Record;

        const ComplexVec e_ref = recieve_signal(a, quiet, frac_target, wave_1, 0.0, p_analytic);
        const ComplexVec e_lin = recieve_signal(a, quiet, frac_target, wave_1, 0.0, p_linear);
        const ComplexVec e_cub = recieve_signal(a, quiet, frac_target, wave_1, 0.0, p_cubic);

        /// Oversampling is the cheap fix for delay accuracy: at 8x the 0.5-sample lag becomes
        /// 4 fine samples, so even a linear kernel reproduces the pulse edges exactly
        const unsigned os = 8;
        ComplexVec a_os(radar_1.samp * os);
        generate_a_signal(a_os, radar_1, wave_1, os);

        echo_path p_os;
        p_os.source       = TxSource::Record;
        p_os.kernel       = DelayKernel::Linear;
        p_os.u_oversample = os;

        const ComplexVec e_os = recieve_signal(a_os, quiet, frac_target, wave_1, 0.0, p_os);

        /// The echo is always resampled onto the dt grid, so the matched-filter reference must
        /// live on that same grid. Feeding make_reference(a_os, ..., os) would hand the filter an
        /// 8x longer template and turn the correlation peak into a 700-sample plateau.
        const std::size_t pk_os =
            peakPowerIndex(matchedFilter(e_os, ref_chk, MatchFilterMode::Same, false));

        /// Relative L2 distance to the analytic reference: a pure interpolation-quality metric
        const auto rel_err = [](const ComplexVec& x, const ComplexVec& r) {
            double num = 0.0, den = 0.0;
            const std::size_t n = std::min(x.size(), r.size());
            for (std::size_t i = 0; i < n; ++i) {
                num += std::norm(x[i] - r[i]);
                den += std::norm(r[i]);
            }
            return (den > 0.0) ? std::sqrt(num / den) : 0.0;
        };

        const std::size_t pk_ref =
            peakPowerIndex(matchedFilter(e_ref, ref_chk, MatchFilterMode::Same, false));
        const std::size_t pk_lin =
            peakPowerIndex(matchedFilter(e_lin, ref_chk, MatchFilterMode::Same, false));
        const std::size_t pk_cub =
            peakPowerIndex(matchedFilter(e_cub, ref_chk, MatchFilterMode::Same, false));

        std::cout << "\npath cross-check: R = " << frac_target.distance
                  << " m, no noise, n_tau = "
                  << round_trip_delay(frac_target.distance) / radar_1.dt << " samples\n"
                  << "  analytic       peak sample = " << pk_ref << "\n"
                  << "  record linear  peak sample = " << pk_lin
                  << "   rel err = " << rel_err(e_lin, e_ref) << "\n"
                  << "  record cubic   peak sample = " << pk_cub
                  << "   rel err = " << rel_err(e_cub, e_ref) << "\n"
                  << "  record linear x" << os << " peak sample = " << pk_os
                  << "   rel err = " << rel_err(e_os, e_ref) << "\n"
                  << std::endl;
    }

    return 0;
}
