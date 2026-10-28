#include "filter.h"

#include <vector>
#include <complex>
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>

using Complex    = std::complex<double>;
using ComplexVec = std::vector<Complex>;

// 输出模式


// -----------------------------------------------------------------
// 匹配滤波器：y = signal ⊛ h,  h[k] = conj(ref[M-1-k])
// 参数：
//   signal                输入复信号
//   ref                   参考(模板)复信号
//   mode                  输出模式
//   normalizeByRefEnergy  为 true 时用 ref 总能量归一化(峰值最大化为 1)
// 返回：
//   滤波后的复信号
// -----------------------------------------------------------------
ComplexVec matchedFilter(const ComplexVec& signal,
                         const ComplexVec& ref,
                         MatchFilterMode mode,
                         bool normalizeByRefEnergy)
{
    const std::size_t N = signal.size();
    const std::size_t M = ref.size();
    if (N == 0 || M == 0) return {};

    // 直接时域卷积：y_full[n] = Σ_k signal[k] * conj(ref[M-1-(n-k)])
    ComplexVec full(N + M - 1, Complex(0.0, 0.0));
    for (std::size_t n = 0; n < N; ++n) {
        const Complex xn = signal[n];
        if (xn == Complex(0.0, 0.0)) continue;          // 稀疏信号优化
        for (std::size_t k = 0; k < M; ++k)
            full[n + k] += xn * std::conj(ref[M - 1 - k]);
    }

    // 可选：按参考信号能量归一化，使完全匹配时峰值 = 1
    if (normalizeByRefEnergy) {
        double e = 0.0;
        for (const auto& c : ref) e += std::norm(c);   // |c|^2
        if (e > 0.0) {
            const double inv = 1.0 / e;
            for (auto& c : full) c *= inv;
        }
    }

    switch (mode) {
    case MatchFilterMode::Full:
        return full;

    case MatchFilterMode::Same: {
        // 取 full[M-1 : M-1+N]，索引 t 对应信号从 t 起与 ref 匹配
        ComplexVec out(N);
        std::copy(full.begin() + static_cast<std::ptrdiff_t>(M - 1),
                  full.begin() + static_cast<std::ptrdiff_t>(M - 1 + N),
                  out.begin());
        return out;
    }

    case MatchFilterMode::Valid: {
        if (N < M) return {};                          // 无完全重叠区
        const std::size_t L = N - M + 1;
        ComplexVec out(L);
        std::copy(full.begin() + static_cast<std::ptrdiff_t>(M - 1),
                  full.begin() + static_cast<std::ptrdiff_t>(M - 1 + L),
                  out.begin());
        return out;
    }
    }
    return {};
}



// -----------------------------------------------------------------
// 信号功率测量
// -----------------------------------------------------------------

// 总能量：E = Σ |x[n]|^2
double signalEnergy(const ComplexVec& x)
{
    double e = 0.0;
    for (const auto& c : x) e += std::norm(c);         // |c|^2
    return e;
}

// 峰值功率：P_peak = max |x[n]|^2
double peakPower(const ComplexVec& x)
{
    double p = 0.0;
    for (const auto& c : x) p = std::max(p, std::norm(c));
    return p;
}

// 峰值功率对应的索引；空信号返回 0
std::size_t peakPowerIndex(const ComplexVec& x)
{
    std::size_t idx = 0;
    double best = -1.0;
    for (std::size_t n = 0; n < x.size(); ++n) {
        const double p = std::norm(x[n]);
        if (p > best) { best = p; idx = n; }
    }
    return idx;
}

// 功率转 dB：10*log10(P / ref)，P<=0 时返回 -inf
double powerToDB(double power, double ref)
{
    if (power <= 0.0 || ref <= 0.0)
        return -std::numeric_limits<double>::infinity();
    return 10.0 * std::log10(power / ref);
}

// -----------------------------------------------------------------
// 测量信号功率
// 参数：
//   x      输入复信号（时域采样）
//   in_db  为 true 时返回 dB 值（相对 1.0，即 dBW/dBFS）
//   ref    归一化参考功率，默认 1.0
// 返回：
//   平均功率 P = (1/N) * Σ |x[n]|^2；空信号返回 0
// 说明：
//   平均功率 = 总能量 / 采样点数，适用于噪声、回波等整段信号；
//   若只关心匹配滤波输出的峰值，请用 peakPower(x)。
// -----------------------------------------------------------------
double measureSignalPower(const ComplexVec& x,
                          bool in_db,
                          double ref)
{
    if (x.empty()) return 0.0;

    const double p = signalEnergy(x) / static_cast<double>(x.size());
    return in_db ? powerToDB(p, ref) : p;
}

// -----------------------------------------------------------------
// 离散 PI 控制器（位置式 + 积分分离 + 抗饱和）
//   u[k] = kp * e[k] + Σ_i (ki * e[i])
// 抗积分饱和：
//   1) 积分分离：|e| 大于 integ_gate 时不做积分，只用比例（大偏差快速拉回，
//      避免乘法型被控对象在远距离目标上把积分冲爆导致发散）；
//   2) 输出饱和且误差仍朝饱和方向推动时，冻结积分。
// -----------------------------------------------------------------
double PIController::update(double error)
{
    double u = kp * error + integral;                  // 位置式 PI
    u = std::clamp(u, out_min, out_max);

    const bool sat_high  = (u >= out_max) && (error > 0.0);
    const bool sat_low   = (u <= out_min) && (error < 0.0);
    const bool gate_open = std::abs(error) <= integ_gate;   // 积分分离
    if (gate_open && !sat_high && !sat_low)
        integral += ki * error;                        // 只在未饱和时积分

    return u;
}

// -----------------------------------------------------------------
// 功率闭环放大器（离散 PI）
// 被控对象：y[n] = g * x[n]  =>  平均功率 P = g^2 * P_in（关于 g 非线性）
// 控制律：
//   e     = P_target - P_meas                       (功率差, W)
//   e_rel = e / P_meas                              (归一化, 无量纲)
//   u     = kp * e_rel + Σ (ki * e_rel)             (PI 输出, 无量纲)
//   期望功率 P_des = P_meas * (1 + u)
// 由于功率 ∝ g^2，对应增益缩放 s = sqrt(1 + u)，故 g <- g * s。
// 当 kp = 1 时 P_des = P_meas + e = P_target，可单步命中目标；
// 积分项用于消除残差并适应模型误差。
//
// 功率感知：
//   Analytic  标量增益下 P(g) = g^2 * P_in，P_in 只在循环外算一次，
//             每轮 O(1)；输出信号仅在收敛后重建一次。
//   Measured  每轮重建 y 并遍历求功率，O(N)。为频域响应 H(f)、
//             饱和/限幅、AM-PM 等非标量模型预留的"真实测量"路径。
// -----------------------------------------------------------------
PowerLoopReport amplifyToTargetPower(const ComplexVec& input, double target_power)
{
    return amplifyToTargetPower(input, target_power, PowerLoopConfig{});
}

PowerLoopReport amplifyToTargetPower(const ComplexVec& input,
                                     double target_power,
                                     const PowerLoopConfig& cfg)
{
    const double kp           = cfg.kp;
    const double ki           = cfg.ki;
    const int    max_iter     = cfg.max_iter;
    const double initial_gain = cfg.initial_gain;
    const double max_gain     = cfg.max_gain;
    std::vector<double>* const history = cfg.history;   // 指向外部容器，可为空
    PowerLoopReport rep;
    rep.output = input;                    // 未收敛/异常时退化为直通
    rep.gain   = initial_gain;

    if (input.empty() || target_power <= 0.0 || max_iter <= 0 || max_gain <= 0.0)
        return rep;

    // 输入功率只需算一次：标量增益下 P_out = g^2 * P_in
    const double p_in = measureSignalPower(input);

    // 全零输入无法被放大到任意功率
    if (p_in <= 0.0) {
        rep.power     = 0.0;
        rep.abs_error = target_power;
        rep.rel_error = 1.0;               // 实际功率为 0，相对误差 100%
        return rep;
    }

    PIController pi;
    pi.kp         = kp;
    pi.ki         = ki;
    pi.integ_gate = cfg.integ_gate;
    pi.reset();

    double g = (initial_gain > 0.0) ? std::min(initial_gain, max_gain) : 1.0;

    // 放大器：y = g * x（Measured 模式下每轮都要真实重建）
    const auto amplify = [&input](double gain) {
        ComplexVec y(input.size());
        for (std::size_t n = 0; n < input.size(); ++n)
            y[n] = input[n] * gain;
        return y;
    };

    double p_meas = 0.0;

    for (int k = 0; k < max_iter; ++k) {
        // 1) 放大器作用 + 2) 测量实际功率
        if (cfg.sense == PowerSenseMode::Analytic) {
            p_meas = g * g * p_in;                 // O(1)，不解向量
        } else {
            p_meas = measureSignalPower(amplify(g)); // O(N)，真实测量
        }
        if (history) history->push_back(p_meas);   // 注：history 为外部容器指针

        // 3) 理想功率 - 实际功率
        const double err = target_power - p_meas;
        const double rel = std::abs(err) / target_power;

        rep.iterations = k + 1;
        rep.gain       = g;
        rep.power      = p_meas;
        rep.abs_error  = err;
        rep.rel_error  = rel;

        // 4) 收敛判据：相对误差小于阈值
        if (rel < cfg.tol_rel) {
            rep.converged = true;
            break;
        }

        // 5) 误差代入 PI，得到功率修正比例 u
        const double p_ref = std::max(p_meas, target_power * 1.0e-12); // 防止除零
        const double e_rel = err / p_ref;              // 归一化功率误差
        const double u     = pi.update(e_rel);

        // 6) 功率修正 -> 增益修正（功率 ∝ g^2，故取 sqrt）
        const double scale = std::sqrt(std::max(0.0, 1.0 + u));
        g = std::min(g * scale, max_gain);
    }

    // 输出信号只在收尾时重建一次（标量增益下与逐轮重建完全等价）
    rep.gain   = g;
    rep.output = amplify(g);
    rep.power  = p_in * g * g;

    return rep;
}
