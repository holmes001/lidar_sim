/**
 * @file    main.cpp
 * @brief   演示：发射 → 回波 → 功率闭环放大 → 匹配滤波测距
 *
 * @details 流程：generate_a_signal 生成复包络 → recieve_signal 合成含噪回波 →
 *          amplifyToTargetPower 闭环放大到目标功率 → matchedFilter 做脉冲压缩 →
 *          峰值位置换算距离。各中间结果分别存图到 figs/。
 */
#include "transport.h"      ///< radar / object / waveform、波形生成与回波
#include "filter.h"         ///< 匹配滤波
#include "doppler.h"        ///< range-Doppler
#include "power.h"          ///< 功率测量与闭环放大
#include "app.h"            ///< FFT / 取样 / 坐标轴 / 存图
#include "plot.h"           ///< 绘图（pl::Style / cv::Mat）

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

/// 图片文件名（相对 figs/ 目录），完整路径由 save_picture 拼接
const std::string fig_pulse = "pulse_with_error.png";
const std::string fig_pid   = "pid.png";
const std::string fig_mf    = "matched_filter.png";
const std::string fig_tx    = "tx_waveform.png";
const std::string fig_rd    = "range_doppler.png";
const std::string fig_rd_r  = "rd_range_cut.png";
const std::string fig_rd_v  = "rd_doppler_cut.png";

int main()
{
    radar radar_1;
    radar_1.rx_area = 1;                 ///< 接收面积 [m^2]
    radar_1.tx_area = 1;                 ///< 发射面积 [m^2]
    radar_1.samp = 24000;                ///< 采样点数；最大不模糊距离 = samp·dt·c/2 = 3600 m
    radar_1.dt = 1e-9;                   ///< 采样间隔；距离分辨率 c·dt/2 = 0.15 m
    radar_1.wavelenth = 0.03;            ///< 中心波长 [m]
    radar_1.tx_p = 1.0;                  ///< 发射功率 [W]
    radar_1.height = 1000;               ///< 高度 [m]
    radar_1.noise_sigma = 2e-8;          ///< 每个 I/Q 分量的噪声标准差

    waveform wave_1;
    wave_1.kind = WF_PULSE;
    wave_1.peak_amp = sqrt(radar_1.tx_p);

    wave_1.param.pulse.width_s = 100*radar_1.dt;   ///< 脉宽 1 个采样点（1 ns）
    wave_1.param.pulse.t_start_s = 0.0;   ///< 脉冲起始时刻 0


    /*
    wave_1.param.impulse.period_s = 1e-6;   ///< 脉冲重复间隔 1 µs
    wave_1.param.impulse.t_start_s = 0.0;   ///< 每周期最多一个非零样点，起始时刻 0
    */
    /*
    wave_1.param.fmcw.t_start_s    = 0.0;
    wave_1.param.fmcw.sweep_s      = 1e-6;    ///< 扫频时长 1 µs
    wave_1.param.fmcw.bandwidth_hz = 100e6;   ///< 带宽 100 MHz → 距离分辨率 c/2B = 1.5 m
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

    /// 匹配滤波参考模板：按波形种类截取一个码元
    ComplexVec ref = make_reference(a, wave_1, radar_1);

    /// Same 模式：输出与输入等长，索引 t 即"从第 t 个采样点开始匹配"，
    /// 因此峰值位置直接就是回波到达时刻对应的采样点
    ComplexVec mf = matchedFilter(amplified, ref, MatchFilterMode::Same, false);

    std::vector<double> mf_mag = vector_abs(mf);

    /// 基带包络实部（generate_a_signal 的输出）与闭环放大后的回波实部
    std::vector<double> tx_real = vector_real(a);
    std::vector<double> rx_real = vector_real(amplified);

    /// 时域图横轴用时间 [µs]；匹配滤波图横轴用时延换算的距离 [m]
    const std::vector<double> t_us = time_axis(radar_1.samp, radar_1.dt);
    const std::vector<double> r_m  = rangeAxis(radar_1.samp, radar_1.dt);

    /// 时域图按时间窗裁剪：发射窗只有 1 µs，回波落在 20 µs 附近，整段 24 µs 画不下
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

    /// 功率闭环收敛曲线：横轴本身就是迭代序号，0..N-1 即迭代次数，仍用 PlotLine
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

    /// 峰值位置 → 目标距离（距离 = 采样点 · c·dt/2）
    const std::size_t pk = peakPowerIndex(mf);
    std::cout << "matched filter peak at sample " << pk
              << "  =>  range = " << pk * radar_1.dt * c / 2.0 << " m\n"
              << "peak/mean power = "
              << powerToDB(peakPower(mf), measureSignalPower(mf)) << " dB"
              << std::endl;

    /// ---- range-Doppler：脉冲串 + 慢时间 FFT ----
    /// 帧参数集中定义；行号由发射时序 t = p·PRI 决定，不做峰值检测。
    pulse_doppler_frame frame;
    frame.n_pulse = 64;        ///< 每帧脉冲数 N，多普勒分辨率 = 1/(N·PRI) = 156 Hz
    frame.pri     = 100e-6;    ///< 脉冲重复间隔 [s]，不模糊距离 = c·PRI/2 = 15 km

    ComplexVec2D pulses = recieve_frame(a, radar_1, object_1, wave_1, frame);

    /// 下游画图与打印沿用这两个别名
    const int    n_pulse = frame.n_pulse;
    const double pri     = frame.pri;

    /// 快时间：逐脉冲匹配滤波 → 慢时间矩阵 slow[p][r]
    ComplexVec2D slow = matchedFilterFrame(pulses, ref, MatchFilterMode::Same);

    /// 慢时间：对每个距离门做 N 点 FFT → range-Doppler 图 rd[k][r]
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

    /// 相邻脉冲在峰值距离门上的相位差，应等于 2π·fd·PRI
    const std::size_t gate = peak_rd.range_index;
    if (slow.size() > 1 && gate < slow[1].size())
    {
        const double dphi = std::arg(slow[1][gate] * std::conj(slow[0][gate]));
        std::cout << "  phase step between pulses = " << dphi << " rad"
                  << "   (theory 2*pi*fd*PRI = " << 2.0 * kpi * fd_theory * pri << " rad)\n";
    }

    /// range-Doppler 图的颜色用 dB（相对全局峰值）：线性幅度下 sinc 旁瓣与噪声底
    /// 会被压成死黑，图上只剩一个亮点。
    const double ref_amp = std::sqrt(peak_rd.power);   ///< 全局峰值幅度
    const auto to_db = [ref_amp](const Complex &v) {
        const double a = std::abs(v) / ref_amp;
        return 20.0 * std::log10(std::max(a, 1e-6));   ///< 下限 -120 dB，避免 log(0)
    };

    if (!rd.empty() && !rd[0].empty())
    {
        const std::size_t n_range = rd[0].size();
        const std::vector<double> range_all = rangeAxis(n_range, radar_1.dt);

        /// 热图：只截峰值附近若干距离门，否则 24000 列会被压成一条线
        const std::size_t half_win = 120;
        const std::size_t r0 = (peak_rd.range_index > half_win)
                             ? peak_rd.range_index - half_win : 0;
        const std::size_t r1 = std::min(n_range, peak_rd.range_index + half_win + 1);

        std::vector<std::vector<double>> rd_db(n_pulse, std::vector<double>(r1 - r0));
        for (int k = 0; k < n_pulse; ++k)
            for (std::size_t r = r0; r < r1; ++r)
                rd_db[k][r - r0] = to_db(rd[k][r]);

        pl::Style st_rd;
        /// 注意：伪彩色图的 y 轴标签是水平居中画在 margin_left 处（不旋转），
        /// 标签过长会向左溢出被裁，故此处只写"doppler bin"，单位放进标题。
        st_rd.title    = "Range-Doppler (dB re peak, 156 Hz/bin)";
        st_rd.xlabel   = "range bin (0.15 m each)";
        st_rd.ylabel   = "doppler bin";
        st_rd.auto_y   = false;   ///< 伪彩色图里 y_min/y_max 被当作色标 vmin/vmax
        st_rd.y_min    = -60.0;
        st_rd.y_max    = 0.0;
        st_rd.decimals = 0;
        save_figure(pl::PlotPseudoColor(pl::Make2D(rd_db), st_rd,
                                        cv::COLORMAP_TURBO, true), fig_rd);

        /// 距离剖面：固定峰值多普勒门，横轴用物理距离。
        /// 跨度别太大：0.15 m/点 与像素宽度的采样比接近 1 时才不会出现混叠波纹。
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

        /// 多普勒谱：固定峰值距离门，横轴用径向速度
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
        st_cut_v.show_markers = true;   ///< 只有 N 个点，标出来更易读
        save_figure(pl::PlotLineXY(pl::Make1D(v_axis), pl::Make1D(cut_v), st_cut_v),
                    fig_rd_v);
    }

    return 0;
}
