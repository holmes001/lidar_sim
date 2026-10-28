/**
 * @file    power.cpp
 * @brief   信号功率测量与功率闭环放大实现
 */
#include "power.h"

#include <algorithm>
#include <cmath>
#include <limits>

/**
 * @name 信号功率测量
 * @{
 */

/// 总能量：E = Σ |x[n]|²
double signalEnergy(const ComplexVec& x)
{
    double e = 0.0;
    for (const auto& c : x) e += std::norm(c);
    return e;
}

/// 峰值功率：P_peak = max |x[n]|²
double peakPower(const ComplexVec& x)
{
    double p = 0.0;
    for (const auto& c : x) p = std::max(p, std::norm(c));
    return p;
}

/// 峰值功率对应的索引；空信号返回 0
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

/// 功率转 dB：10·log10(P / ref)，P<=0 时返回 -inf
double powerToDB(double power, double ref)
{
    if (power <= 0.0 || ref <= 0.0)
        return -std::numeric_limits<double>::infinity();
    return 10.0 * std::log10(power / ref);
}

/**
 * @brief 平均功率 P = (1/N)·Σ|x[n]|²
 * @param x     输入复信号
 * @param in_db 为 true 时返回 dB 值（相对 1.0，即 dBW/dBFS）
 * @param ref   dB 归一化参考功率
 * @return 平均功率；空信号返回 0
 * @note 平均功率 = 总能量 / 采样点数，适用于噪声、回波等整段信号；
 *       只关心匹配滤波输出峰值时请用 peakPower。
 */
double measureSignalPower(const ComplexVec& x,
                          bool in_db,
                          double ref)
{
    if (x.empty()) return 0.0;

    const double p = signalEnergy(x) / static_cast<double>(x.size());
    return in_db ? powerToDB(p, ref) : p;
}

/** @} */

/**
 * @brief 离散 PI 单步更新
 * @param error 归一化误差（无量纲）
 * @return 限幅后的控制量
 *
 * @note 抗积分饱和有三条：
 *       1) 积分分离——|e| 大于 integ_gate 时不做积分，只用比例；
 *       2) 输出已达上限且误差仍为正时冻结积分；
 *       3) 输出已达下限且误差仍为负时冻结积分。
 */
double PIController::update(double error)
{
    double u = kp * error + integral;                  ///< 位置式 PI
    u = std::clamp(u, out_min, out_max);

    const bool sat_high  = (u >= out_max) && (error > 0.0);
    const bool sat_low   = (u <= out_min) && (error < 0.0);
    const bool gate_open = std::abs(error) <= integ_gate;   ///< 积分分离
    if (gate_open && !sat_high && !sat_low)
        integral += ki * error;                        ///< 仅在未饱和时积分

    return u;
}

/// 功率闭环放大：默认参数版本
PowerLoopReport amplifyToTargetPower(const ComplexVec& input, double target_power)
{
    return amplifyToTargetPower(input, target_power, PowerLoopConfig{});
}

/**
 * @brief 功率闭环放大：迭代调节增益 g 使输出平均功率趋于目标
 * @param input        输入复信号
 * @param target_power 理想输出功率
 * @param cfg          闭环参数
 * @return 迭代报告
 *
 * @note 被控对象 y[n] = g·x[n]，平均功率 P = g²·P_in，对 g 非线性；
 *       因此误差在功率上归一化，修正量再开方后作用于增益：
 *       e = P_target - P_meas，e_rel = e/P_meas，u = PI(e_rel)，
 *       P_des = P_meas·(1+u)，g ← g·sqrt(1+u)。
 *       kp = 1 时 P_des = P_target，可单步命中；积分项消除残差。
 * @note Analytic 模式下 P_in 只在循环外算一次，每轮 O(1)，输出仅在收尾时重建；
 *       Measured 模式每轮重建 y 并遍历求功率，O(N)，供非标量模型使用。
 */
PowerLoopReport amplifyToTargetPower(const ComplexVec& input,
                                     double target_power,
                                     const PowerLoopConfig& cfg)
{
    const double kp           = cfg.kp;
    const double ki           = cfg.ki;
    const int    max_iter     = cfg.max_iter;
    const double initial_gain = cfg.initial_gain;
    const double max_gain     = cfg.max_gain;
    std::vector<double>* const history = cfg.history;   ///< 指向外部容器，可为空
    PowerLoopReport rep;
    rep.output = input;                    ///< 未收敛/异常时退化为直通
    rep.gain   = initial_gain;

    if (input.empty() || target_power <= 0.0 || max_iter <= 0 || max_gain <= 0.0)
        return rep;

    /// 输入功率只需算一次：标量增益下 P_out = g²·P_in
    const double p_in = measureSignalPower(input);

    /// 全零输入无法被放大到任意功率
    if (p_in <= 0.0) {
        rep.power     = 0.0;
        rep.abs_error = target_power;
        rep.rel_error = 1.0;               ///< 实际功率为 0，相对误差 100%
        return rep;
    }

    PIController pi;
    pi.kp         = kp;
    pi.ki         = ki;
    pi.integ_gate = cfg.integ_gate;
    pi.reset();

    double g = (initial_gain > 0.0) ? std::min(initial_gain, max_gain) : 1.0;

    /// 放大器：y = g·x（Measured 模式下每轮都要真实重建）
    const auto amplify = [&input](double gain) {
        ComplexVec y(input.size());
        for (std::size_t n = 0; n < input.size(); ++n)
            y[n] = input[n] * gain;
        return y;
    };

    double p_meas = 0.0;

    for (int k = 0; k < max_iter; ++k) {
        /// 1) 放大器作用 + 2) 测量实际功率
        if (cfg.sense == PowerSenseMode::Analytic) {
            p_meas = g * g * p_in;                 ///< O(1)，不解向量
        } else {
            p_meas = measureSignalPower(amplify(g)); ///< O(N)，真实测量
        }
        if (history) history->push_back(p_meas);

        /// 3) 理想功率 - 实际功率
        const double err = target_power - p_meas;
        const double rel = std::abs(err) / target_power;

        rep.iterations = k + 1;
        rep.gain       = g;
        rep.power      = p_meas;
        rep.abs_error  = err;
        rep.rel_error  = rel;

        /// 4) 收敛判据：相对误差小于阈值
        if (rel < cfg.tol_rel) {
            rep.converged = true;
            break;
        }

        /// 5) 误差代入 PI，得到功率修正比例 u
        const double p_ref = std::max(p_meas, target_power * 1.0e-12); ///< 防止除零
        const double e_rel = err / p_ref;              ///< 归一化功率误差
        const double u     = pi.update(e_rel);

        /// 6) 功率修正 → 增益修正（功率 ∝ g²，故取 sqrt）
        const double scale = std::sqrt(std::max(0.0, 1.0 + u));
        g = std::min(g * scale, max_gain);
    }

    /// 输出信号只在收尾时重建一次（标量增益下与逐轮重建等价）
    rep.gain   = g;
    rep.output = amplify(g);
    rep.power  = p_in * g * g;

    return rep;
}
