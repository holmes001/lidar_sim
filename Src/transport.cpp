#include "transport.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <complex>

#include "random_error.h"

/* 已废弃：噪声不再按回波能量缩放，改由 radar::noise_sigma 决定。
 * 保留声明只为不破坏可能存在的旧引用，不参与任何计算。 */
double noise_factor = 0.1;

/* 随机种子：每调用一次 recieve_signal 递增。
 * 这样同一次运行里多个目标的噪声互不相关，而整次运行仍可完全复现。 */
unsigned int g_seed = 8051;

// 纯物理公式：只依赖数值，便于单测和复用
double decay_factor(double tx_area, double rx_area, double obj_area,
                    double wavelength, double distance)
{
    const double R2 = distance * distance;
    const double R4 = R2 * R2;
    const double lambda2 = wavelength * wavelength;
    return (rx_area * tx_area * obj_area) / (4.0 * kpi * lambda2 * R4);
}

double delay(double R)
{
    return R / c;
}

double round_trip_delay(double R)
{
    return 2.0 * R / c;
}

/* 波形在连续时刻 t 的实幅度；dt 为采样间隔（冲激串用它判断“是否命中采样点”）。
 * 脉冲窗取半开区间 [t_start, t_start+width)：端点正好落在采样点上时，
 * 开/闭一致，不会因为浮点末位差异让发射和回波的脉冲宽度差一个样点。 */
static double wf_eval(const waveform &w, double t, double dt)
{
    if (w.kind == WF_PULSE)
    {
        const double t_start = w.param.pulse.t_start_s;
        const double tau     = w.param.pulse.width_s;
        return (t >= t_start && t < t_start + tau) ? w.peak_amp : 0.0;
    }
    if (w.kind == WF_SINE)
    {
        return w.peak_amp * std::sin(2.0 * kpi / w.param.sine.period_s * t);
    }
    if (w.kind == WF_IMPULSE)
    {
        const double t_start = w.param.impulse.t_start_s;
        const double period  = w.param.impulse.period_s;
        if (period <= 0.0 || dt <= 0.0) return 0.0;

        const double d    = t - t_start;
        const double half = 0.5 * dt;
        const double eps  = 1.0e-6 * dt;   // 吸收浮点舍入，避免半样点边界上漏掉冲激

        /* 到最近一个冲激时刻的有符号偏移。只有偏移落在 [-dt/2, dt/2) 才认为该
         * 采样点砸在冲激上：半开区间保证正好卡在两个采样点中间时也只出一个
         * 非零样点，与脉冲窗的开/闭约定一致。
         * 注意下界带容差：回波按连续时延 (i-ntau)*dt 求值，若时延是分数样点，
         * 离第一个冲激最近的采样点可能略早于 t_start，仍应判为命中。 */
        const double k   = std::floor(d / period + 0.5);
        const double off = d - k * period;
        return (off >= -half - eps && off < half - eps) ? w.peak_amp : 0.0;
    }
    return 0.0;
}

void generate_a_signal(std::vector<std::complex<double>> &u,const radar &radar_1,waveform wave_1)
{
    const std::size_t n = std::min(u.size(), static_cast<std::size_t>(radar_1.samp));
    for(std::size_t i=0;i<n;i++)
    {
        // 统一在连续时间上求值：冲激串由此自动变成“每个周期最多一个采样点非零”
        u[i] = wf_eval(wave_1, static_cast<double>(i) * radar_1.dt, radar_1.dt);
    }
}


std::vector<std::complex<double>> recieve_signal(
    const std::vector<std::complex<double>> &u,const radar &radar_1,const object &object_1,
    const waveform &wave)
{
    (void)u;   // 回波改为按 wave 解析求值，不再从 u 里搬移采样点
    std::vector<std::complex<double>> v(static_cast<std::size_t>(radar_1.samp));

    // 雷达方程给的是功率比，取平方根才是幅度因子
    const double A = std::sqrt(
        decay_factor(radar_1.tx_area, radar_1.rx_area, object_1.area,
                     radar_1.wavelenth, object_1.distance));

    // 往返时延是连续量：直接在 t = (i - tau/dt)*dt 处对波形求值。
    // 不写成 i*dt - tau，是因为在 i*dt 与 tau 都约为 2e-5 s 时相减会损失两位有效数字，
    // 边界上会与发射端的 t = i*dt 判断不一致（多发/少发一个样点）。
    const double tau    = round_trip_delay(object_1.distance);
    const double n_tau  = tau / radar_1.dt;          // 分数样点延迟
    const double fd = 2.0 * object_1.velocity / radar_1.wavelenth;   // 多普勒 Hz
    for(int i = 0; i < radar_1.samp; i++)
    {
        const double t = (static_cast<double>(i) - n_tau) * radar_1.dt;
        v[static_cast<std::size_t>(i)] = A * wf_eval(wave, t, radar_1.dt) * std::exp(std::complex<double>(0.0, 2.0 * kpi * fd * t));
    }

    // 噪声只由接收机决定：noise_sigma 与距离、目标是否存在都无关。
    // 逐次递增 g_seed，保证同一进程内多个目标的噪声互不相关。
    return addGaussianNoiseStd(v, radar_1.noise_sigma, 0.0, g_seed++);
    //return v;
}
