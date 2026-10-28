/**
 * @file    main.cpp
 * @brief
 *
 * @details
 *
 * @section
 * @note
 */

#include "transport.h"
#include "random_error.h"
#include "plot.h"
#include "filter.h"


#include <cmath>
#include <complex>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

// 图片文件名（相对 figs/ 目录），实际路径由 save_picture 拼接
const std::string fig_pulse = "pulse_with_error.png";
const std::string fig_pid   = "pid.png";
const std::string fig_mf    = "matched_filter.png";
const std::string fig_tx    = "tx_waveform.png";

namespace pl = lidar::plot;

// 图输出目录：以本文件(main.cpp)位置为基准回到工程根，再拼 figs/。
// 因此无论从 build/ 还是从工程根启动，图片都落到 <工程根>/figs/ 下。
static std::filesystem::path fig_dir()
{
    const std::filesystem::path self(__FILE__);        // .../Src/main.cpp
    return self.parent_path().parent_path() / "figs";
}

std::vector<double> vector_real(const std::vector<std::complex<double>> &u)
{
    std::vector<double> v(u.size());
    for (std::size_t i = 0; i < u.size(); ++i)
    {
        v[i] = std::real(u[i]);
    }
    return v;
}

std::vector<double> vector_abs(const std::vector<std::complex<double>> &u)
{
    std::vector<double> v(u.size());
    for (std::size_t i = 0; i < u.size(); ++i)
    {
        v[i] = std::abs(u[i]);
    }
    return v;
}

// 从发射记录里截出"一个码元"作为匹配滤波参考：
//   WF_PULSE   -> 非零段（矩形脉冲本身）
//   WF_IMPULSE -> 一个周期（含单个冲激样点）
//   WF_SINE    -> 一个周期；周期长于记录时退化为记录的前 1/8
static ComplexVec make_reference(const std::vector<std::complex<double>> &tx,
                                 const waveform &w, const radar &r)
{
    ComplexVec ref;
    if (tx.empty()) return ref;

    if (w.kind == WF_PULSE)
    {
        for (const auto &v : tx)
            if (std::abs(v) > 0.0) ref.push_back(v);
        return ref;
    }

    const double period = (w.kind == WF_SINE)    ? w.param.sine.period_s
                        : (w.kind == WF_IMPULSE) ? w.param.impulse.period_s
                                                 : w.param.square.period_s;
    std::size_t len = static_cast<std::size_t>(std::llround(period / r.dt));
    const std::size_t cap = std::max<std::size_t>(2, tx.size() / 8);
    len = std::min(std::max<std::size_t>(len, 2), cap);
    len = std::min(len, tx.size());

    ref.assign(tx.begin(), tx.begin() + static_cast<std::ptrdiff_t>(len));
    return ref;
}

// 第三个参数为保存路径：
//   绝对路径 -> 直接用；
//   相对路径 -> 相对工程根的 figs/ 目录(如 "pulse.png" => <工程根>/figs/pulse.png)；
//               目录不存在会自动创建。
bool save_picture(const cv::Mat &a, const pl::Style &st, const std::string &file_path)
{
    std::filesystem::path out(file_path);
    if (out.is_relative())
        out = fig_dir() / out;

    std::error_code ec;
    std::filesystem::create_directories(out.parent_path(), ec);

    cv::Mat fig = pl::PlotLine(a, st);
    if (pl::Save(fig, out.string()))
    {
        std::cout << "picture saved in " << out.string() << std::endl;
        return true;
    }
    std::cout << "save failed: " << out.string() << std::endl;
    return false;
}

int main()
{
    radar radar_1;
    radar_1.rx_area = 1;                 // m^2
    radar_1.tx_area = 1;                 // m^2
    radar_1.samp = 24000;                // 最大不模糊距离 = samp*dt*c/2 = 3600 m
    radar_1.dt = 1e-9;                   // 1 ns，距离分辨率 c*dt/2 = 0.15 m
    radar_1.wavelenth = 0.03;            //
    radar_1.tx_p = 1.0;                  // W
    radar_1.height = 1000;               // m
    radar_1.noise_sigma = 2e-8;          // 每个 I/Q 分量的接收机噪声标准差

    waveform wave_1;
    wave_1.kind = WF_SINE;
    wave_1.peak_amp = sqrt(radar_1.tx_p);

    wave_1.param.impulse.period_s = 1e-5;
    wave_1.param.impulse.t_start_s = 200e-9;
    wave_1.param.pulse.t_start_s = 200e-9;   // 发射窗起点 200 ns
    wave_1.param.pulse.width_s  = 10e-9;     // 脉宽 10 ns
    wave_1.param.sine.period_s = 1e-5;


    object object_1;
    object_1.area = 0.2;
    object_1.distance = 3000;
    object_1.theta = kpi;
    object_1.velocity = 20;

    std::vector<std::complex<double>> a(radar_1.samp);

    std::vector<std::complex<double>> rs_1(radar_1.samp);
    std::vector<std::complex<double>> rs_2(radar_1.samp);
    std::vector<std::complex<double>> rs_3(radar_1.samp);


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

    // 匹配滤波参考模板：按波形种类截取"一个码元"
    ComplexVec ref = make_reference(a, wave_1, radar_1);

    // Same 模式：输出与输入等长，索引 t 即"从第 t 个采样点开始匹配"，
    // 因此峰值位置直接就是回波到达时刻对应的采样点
    ComplexVec mf = matchedFilter(amplified, ref, MatchFilterMode::Same, false);

    std::vector<double> mf_mag = vector_abs(mf);

    // 发射波形（最开始 generate_a_signal 产生的信号），留作与回波对比
    std::vector<double> tx_real = vector_real(a);

    std::vector<std::vector<double>> re(3,std::vector<double>(radar_1.samp));
    re[0] = vector_real(amplified);

    cv::Mat m_tx = pl::Make1D(tx_real);
    cv::Mat m_0  = pl::Make1D(re[0]);
    cv::Mat m_1  = pl::Make1D(hist);
    cv::Mat m_2  = pl::Make1D(mf_mag);

    // 发射波形
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

    // 匹配滤波输出（取模值）
    pl::Style st_2;
    st_2.title  = "Matched Filter Output";
    st_2.xlabel = "Sample";
    st_2.ylabel = "|y|";
    st_2.legend = {"matched filter"};
    save_picture(m_2, st_2, fig_mf);

    // 峰值位置 -> 目标距离标定（距离 = 采样点 * c*dt/2）
    const std::size_t pk = peakPowerIndex(mf);
    std::cout << "matched filter peak at sample " << pk
              << "  =>  range = " << pk * radar_1.dt * c / 2.0 << " m\n"
              << "peak/mean power = "
              << powerToDB(peakPower(mf), measureSignalPower(mf)) << " dB"
              << std::endl;

    return 0;
}
