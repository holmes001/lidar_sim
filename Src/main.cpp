/**
 * @file    main.cpp
 * @brief   演示：发射 → 回波 → 功率闭环放大 → 匹配滤波测距
 *
 * @details 流程：generate_a_signal 生成复包络 → recieve_signal 合成含噪回波 →
 *          amplifyToTargetPower 闭环放大到目标功率 → matchedFilter 做脉冲压缩 →
 *          峰值位置换算距离。各中间结果分别存图到 figs/。
 */
#include "transport.h"
#include "random_error.h"
#include "plot.h"
#include "filter.h"
#include "power.h"


#include <cmath>
#include <complex>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

/// 图片文件名（相对 figs/ 目录），完整路径由 save_picture 拼接
const std::string fig_pulse = "pulse_with_error.png";
const std::string fig_pid   = "pid.png";
const std::string fig_mf    = "matched_filter.png";
const std::string fig_tx    = "tx_waveform.png";
const std::string fig_rd    = "range_doppler.png";

namespace pl = lidar::plot;

/**
 * @brief 图片输出目录
 * @return <工程根>/figs
 * @note 以本文件路径为基准回退两级，因此无论从 build/ 还是工程根启动，
 *       图片都落到工程根的 figs/ 下。
 */
static std::filesystem::path fig_dir()
{
    const std::filesystem::path self(__FILE__);        ///< .../Src/main.cpp
    return self.parent_path().parent_path() / "figs";
}

/// 取复信号的实部
std::vector<double> vector_real(const ComplexVec &u)
{
    std::vector<double> v(u.size());
    for (std::size_t i = 0; i < u.size(); ++i)
    {
        v[i] = std::real(u[i]);
    }
    return v;
}

/// 取复信号的模
std::vector<double> vector_abs(const ComplexVec &u)
{
    std::vector<double> v(u.size());
    for (std::size_t i = 0; i < u.size(); ++i)
    {
        v[i] = std::abs(u[i]);
    }
    return v;
}

/**
 * @brief 从发射记录中截出一个码元作为匹配滤波参考
 * @param tx 发射复包络
 * @param w  波形参数，决定截取方式
 * @param r  雷达参数，提供 dt
 * @return 参考模板
 *
 * @note 有限长包络（WF_PULSE / WF_FMCW）取包络非零段，此时峰位直接对应回波延时；
 *       周期型包络（WF_SINE / WF_SQUARE / WF_IMPULSE）取一个周期。
 * @warning param 是 union，必须按 kind 读对应成员；跨成员读取属类型双关。
 */
static ComplexVec make_reference(const ComplexVec &tx,
                                 const waveform &w, const radar &r)
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
                    ? static_cast<std::size_t>(std::llround(period / r.dt)) : 0;
    const std::size_t cap = std::max<std::size_t>(2, tx.size() / 8);
    len = std::min(std::max<std::size_t>(len, 2), cap);
    len = std::min(len, tx.size());

    ref.assign(tx.begin(), tx.begin() + len);
    return ref;
}

/**
 * @brief 写盘
 * @param fig       已渲染的图像
 * @param file_path 绝对路径直接用；相对路径视为相对 <工程根>/figs，
 *                  目录不存在会自动创建
 * @return 保存成功返回 true
 */
static bool save_figure(const cv::Mat &fig, const std::string &file_path)
{
    std::filesystem::path out(file_path);
    if (out.is_relative())
        out = fig_dir() / out;

    std::error_code ec;
    std::filesystem::create_directories(out.parent_path(), ec);

    if (pl::Save(fig, out.string()))
    {
        std::cout << "picture saved in " << out.string() << std::endl;
        return true;
    }
    std::cout << "save failed: " << out.string() << std::endl;
    return false;
}

/**
 * @brief 绘图并保存
 * @param a         待绘制的数据矩阵
 * @param st        绘图样式
 * @param file_path 保存路径，规则同 save_figure
 * @return 保存成功返回 true
 */
bool save_picture(const cv::Mat &a, const pl::Style &st, const std::string &file_path)
{
    return save_figure(pl::PlotLine(a, st), file_path);
}

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
    wave_1.kind = WF_FMCW;
    wave_1.peak_amp = sqrt(radar_1.tx_p);

    /// 三种波形的复包络（约定见 transport.h）：
    ///   WF_PULSE  实数包络，矩形窗      → 填 param.pulse.{t_start_s, width_s}
    ///   WF_SINE   常数包络（单频连续波）→ 不填参数，振荡在载波 f₀ = c/λ 里
    ///   WF_FMCW   复包络，瞬时频率 −B/2 → +B/2
    /// param 是 union，只能填当前 kind 对应的成员；t_start_s 必须为 0，
    /// 否则峰位带上 t_start_s 的偏移，测距标定需先减掉。
    wave_1.param.fmcw.t_start_s    = 0.0;
    wave_1.param.fmcw.sweep_s      = 1e-6;    ///< 扫频时长 1 µs
    wave_1.param.fmcw.bandwidth_hz = 100e6;   ///< 带宽 100 MHz → 距离分辨率 c/2B = 1.5 m


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

    /// 发射波形（generate_a_signal 产生的复包络实部），留作与回波对比
    std::vector<double> tx_real = vector_real(a);

    std::vector<std::vector<double>> re(3,std::vector<double>(radar_1.samp));
    re[0] = vector_real(amplified);

    cv::Mat m_tx = pl::Make1D(tx_real);
    cv::Mat m_0  = pl::Make1D(re[0]);
    cv::Mat m_1  = pl::Make1D(hist);
    cv::Mat m_2  = pl::Make1D(mf_mag);

    pl::Style st_tx;
    st_tx.title  = "Transmitted Waveform";
    st_tx.xlabel = "Sample";
    st_tx.ylabel = "Amplitude";
    st_tx.legend = {"tx"};
    save_picture(m_tx, st_tx, fig_tx);

    pl::Style st;
    st.title  = "Pulse";
    st.xlabel = "Sample";
    st.ylabel = "Amplitude";
    st.legend = {"rx_1"};
    save_picture(m_0, st, fig_pulse);

    pl::Style st_1;
    st_1.title  = "Power";
    st_1.xlabel = "iteration";
    st_1.ylabel = "Amplitude";
    st_1.legend = {"error"};
    save_picture(m_1, st_1, fig_pid);

    pl::Style st_2;
    st_2.title  = "Matched Filter Output";
    st_2.xlabel = "Sample";
    st_2.ylabel = "|y|";
    st_2.legend = {"matched filter"};
    save_picture(m_2, st_2, fig_mf);

    /// 峰值位置 → 目标距离（距离 = 采样点 · c·dt/2）
    const std::size_t pk = peakPowerIndex(mf);
    std::cout << "matched filter peak at sample " << pk
              << "  =>  range = " << pk * radar_1.dt * c / 2.0 << " m\n"
              << "peak/mean power = "
              << powerToDB(peakPower(mf), measureSignalPower(mf)) << " dB"
              << std::endl;

    /// ---- range-Doppler：脉冲串 + 慢时间 FFT ----
    const int    n_pulse = 64;        ///< 每帧脉冲数 N
    const double pri     = 100e-6;    ///< 脉冲重复间隔 [s]

    /// 逐脉冲生成回波；t_offset = p·PRI 让多普勒相位跨脉冲连续累进，
    /// 若恒传 0，慢时间序列会变成常量、多普勒恒为 0。
    ComplexVec2D pulses(n_pulse);
    for (int p = 0; p < n_pulse; ++p)
        pulses[p] = recieve_signal(a, radar_1, object_1, wave_1, p * pri);

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

    /// range-Doppler 图：只截峰值附近若干距离门，否则 24000 列会被压成一条线
    if (!rd.empty() && !rd[0].empty())
    {
        const std::size_t half_win = 120;
        const std::size_t r0 = (peak_rd.range_index > half_win)
                             ? peak_rd.range_index - half_win : 0;
        const std::size_t r1 = std::min(rd[0].size(), peak_rd.range_index + half_win + 1);

        std::vector<std::vector<double>> rd_crop(n_pulse, std::vector<double>(r1 - r0));
        for (int k = 0; k < n_pulse; ++k)
            for (std::size_t r = r0; r < r1; ++r)
                rd_crop[k][r - r0] = std::abs(rd[k][r]);

        pl::Style st_rd;
        st_rd.title  = "Range-Doppler";
        st_rd.xlabel = "range bin";
        st_rd.ylabel = "doppler bin";
        save_figure(pl::PlotPseudoColor(pl::Make2D(rd_crop), st_rd,
                                        cv::COLORMAP_TURBO, true), fig_rd);
    }

    return 0;
}
