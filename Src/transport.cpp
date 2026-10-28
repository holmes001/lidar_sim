/**
 * @file    transport.cpp
 * @brief   雷达链路实现：雷达方程、时延、复包络波形与回波合成
 */
#include "transport.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <complex>

#include "random_error.h"

/**
 * @brief 噪声系数（已废弃）
 * @deprecated 噪声改由 radar::noise_sigma 决定；此变量不参与任何计算，
 *             仅为兼容可能存在的旧引用而保留。
 */
double noise_factor = 0.1;

/**
 * @brief 全局随机种子
 * @note 每调用一次 recieve_signal 递增：同一进程内多次回波的噪声互不相关，
 *       重复运行同一份代码结果可复现。
 */
unsigned int g_seed = 8051;

/// 雷达方程衰减因子（只依赖物理量，便于单测与复用）
double decay_factor(double tx_area, double rx_area, double obj_area,
                    double wavelength, double distance)
{
    const double R2 = distance * distance;
    const double R4 = R2 * R2;
    const double lambda2 = wavelength * wavelength;
    return (rx_area * tx_area * obj_area) / (4.0 * kpi * lambda2 * R4);
}

/// 单程时延 R/c [s]
double delay(double R)
{
    return R / c;
}

/// 往返时延 2R/c [s]
double round_trip_delay(double R)
{
    return 2.0 * R / c;
}

/**
 * @brief 复包络在连续时刻 t 的取值
 * @param w  波形参数
 * @param t  时刻 [s]
 * @param dt 采样间隔，仅冲激串用于判断是否命中采样点
 * @return 复包络值；窗口外为 0
 *
 * @note 幅度调制（脉冲、CW）返回实数包络；只有瞬时频率偏离 f₀ 的 FMCW 返回复数。
 * @note 脉冲窗取半开区间，与发射端、回波端一致，避免端点落在样点上时差一个样点。
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
        /// 单频连续波：s(t) = peak_amp·cos(2πf₀t)，故复包络为常数
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
        const double eps  = 1.0e-6 * dt;   ///< 吸收浮点舍入，避免半样点边界漏掉冲激

        /// 只有到最近冲激时刻的偏移落在 [-dt/2, dt/2) 才判为命中；半开区间保证
        /// 正好卡在两样点中间时也只出一个非零样点。下界带容差：分数时延时离第一个
        /// 冲激最近的采样点可能略早于 t_start，仍应判为命中。
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

        const double S = B / T;                            ///< 调频斜率 [Hz/s]
        /// phase = π·S·(tau² − tau·T)，瞬时基带频率 = S·(tau − T/2) ∈ [−B/2, +B/2]
        const double phase = kpi * S * (tau * tau - tau * T);
        return Complex(w.peak_amp * std::cos(phase), w.peak_amp * std::sin(phase));
    }
    }
    return zero;
}

void generate_a_signal(ComplexVec &u,const radar &radar_1,waveform wave_1)
{
    const std::size_t n = std::min(u.size(), radar_1.samp);
    for(std::size_t i=0;i<n;i++)
    {
        /// 统一在连续时间上求值：冲激串由此自动变成"每个周期最多一个采样点非零"
        u[i] = wf_eval(wave_1, static_cast<double>(i) * radar_1.dt, radar_1.dt);
    }
}


ComplexVec recieve_signal(
    const ComplexVec &u,const radar &radar_1,const object &object_1,
    const waveform &wave,double t_offset)
{
    (void)u;   ///< 回波按 wave 解析求值，不再从 u 里搬移采样点
    ComplexVec v(radar_1.samp);

    /// 雷达方程给的是功率比，取平方根才是幅度因子
    const double A = std::sqrt(
        decay_factor(radar_1.tx_area, radar_1.rx_area, object_1.area,
                     radar_1.wavelenth, object_1.distance));

    /// 往返时延是连续量，直接在 t = t_offset + (i − tau/dt)·dt 处求值。
    /// 不写成 i·dt − tau：两者都约为 2e-5 s 时相减会损失两位有效数字，
    /// 边界上会与发射端的 t = i·dt 判断不一致。
    /// t_offset 让多普勒相位跨脉冲累进（第 p 个脉冲传 p·PRI）。
    const double tau    = round_trip_delay(object_1.distance);
    const double n_tau  = tau / radar_1.dt;          ///< 分数样点延迟
    const double fd     = 2.0 * object_1.velocity / radar_1.wavelenth;   ///< 多普勒 [Hz]

    /// 往返载波相位 φ₀ = k·2R = 4πR/λ（对 2π 取模）。多普勒只是它的时间导数，
    /// 常数项必须单独保留：相干积累、干涉测距、多目标相位关系都要用到。
    const double phi0 = std::fmod(2.0 * kpi * (2.0 * object_1.distance / radar_1.wavelenth),
                                  2.0 * kpi);

    for(std::size_t i = 0; i < radar_1.samp; i++)
    {
        /// 包络按"相对本脉冲发射时刻"的滞后时间求值，与 t_offset 无关：
        /// 若把 t_offset 混进来，从第 2 个脉冲起 t 就落到波形窗口之外，回波恒为零。
        const double t_local = (static_cast<double>(i) - n_tau) * radar_1.dt;
        /// 载波相位按**绝对时间**求值：t_offset 正是多普勒相位跨脉冲累进的来源。
        const double t_abs   = t_offset + t_local;
        v[i] = A * wf_eval(wave, t_local, radar_1.dt)
                 * std::exp(Complex(0.0, 2.0 * kpi * fd * t_abs + phi0));
    }

    /// 噪声只由接收机决定，与距离及目标是否存在无关；逐次递增 g_seed 保证不相关
    return addGaussianNoiseStd(v, radar_1.noise_sigma, 0.0, g_seed++);
}
