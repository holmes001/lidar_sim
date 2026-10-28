/**
 * @file    power.h
 * @brief   信号功率测量与功率闭环放大（离散 PI）
 */
#ifndef POWER_H
#define POWER_H

#include <cstddef>

#include "complex_types.h"

/**
 * @name 信号功率测量
 * @{
 */

/** @brief 总能量 E = Σ|x[n]|² */
double      signalEnergy(const ComplexVec& x);

/** @brief 峰值功率 max|x[n]|² */
double      peakPower(const ComplexVec& x);

/** @brief 峰值功率所在的样点索引；空信号返回 0 */
std::size_t peakPowerIndex(const ComplexVec& x);

/**
 * @brief 功率转 dB：10·log10(P / ref)
 * @param power 功率，必须为正
 * @param ref   参考功率，默认 1.0
 * @return dB 值；power 或 ref 非正时返回 -inf
 */
double      powerToDB(double power, double ref = 1.0);

/**
 * @brief 平均功率 P = (1/N)·Σ|x[n]|²
 * @param x     输入复信号
 * @param in_db 为 true 时返回 dB 值
 * @param ref   dB 归一化参考功率
 * @return 平均功率；空信号返回 0
 * @note 适用于噪声、回波等整段信号；只关心匹配滤波峰值时请用 peakPower。
 */
double      measureSignalPower(const ComplexVec& x,
                               bool in_db = false,
                               double ref = 1.0);

/** @} */

/**
 * @brief 离散 PI 控制器（位置式，含积分分离与抗饱和）
 *
 * u[k] = kp·e[k] + Σ_i ki·e[i]。e 为归一化误差、u 为归一化控制量，
 * 二者均无量纲，因此 kp、ki 与信号量纲无关，便于移植。
 *
 * @warning integral 是状态量：换用新的被控对象或重新开始闭环前必须调用 reset()，
 *          否则上一轮积累的积分会带入下一轮。
 */
struct PIController {
    double kp       = 1.0;      ///< 比例增益
    double ki       = 0.2;      ///< 积分增益（每步累积量）
    double out_min  = -0.99;    ///< 控制量下限，保证 1+u > 0
    double out_max  = 1.0e3;    ///< 控制量上限
    double integ_gate = 1.0;    ///< 积分分离阈值：|e| 超过它时只走比例
    double integral = 0.0;      ///< 积分累积项

    /// 单步更新：输入归一化误差，返回限幅后的控制量
    double update(double error);
    /// 清零积分项
    void   reset() { integral = 0.0; }
};

/**
 * @brief 功率闭环放大的迭代报告
 *
 * @note 各字段在一次收敛过程中逐轮刷新，收敛后即为最终值。
 */
struct PowerLoopReport {
    ComplexVec output;             ///< 闭环输出信号 y = g·x
    double     gain       = 1.0;   ///< 最终（总）增益
    double     power      = 0.0;   ///< 输出实际功率
    double     abs_error  = 0.0;   ///< 绝对误差 = 理想功率 - 实际功率
    double     rel_error  = 0.0;   ///< 相对误差 = |abs_error| / 理想功率
    int        iterations = 0;     ///< 实际迭代次数
    bool       converged  = false; ///< 是否满足收敛阈值
};

/** @brief 功率感知方式 */
enum class PowerSenseMode {
    Analytic,   ///< 解析：P = g²·P_in，每轮 O(1)，不重建信号
    Measured    ///< 实测：每轮重建 y 并遍历求功率，O(N)；非标量模型必须用它
};

/**
 * @brief 功率闭环放大器的全部可调参数
 *
 * @warning history 指向外部容器：它必须在本结构体使用期间保持有效，
 *          且不会被扩容失效以外的原因提前销毁。
 */
struct PowerLoopConfig {
    double         kp           = 1.0;    ///< PI 比例增益
    double         ki           = 0.2;    ///< PI 积分增益
    double         integ_gate   = 1.0;    ///< 积分分离阈值：|e_rel| 超过它时只走比例
    double         tol_rel      = 1.0e-3; ///< 收敛阈值：相对误差小于它即停
    int            max_iter     = 100;    ///< 最大迭代次数
    double         initial_gain = 1.0;    ///< 初始增益 g0
    double         max_gain     = 1.0e6;  ///< 增益上限，防止发散
    PowerSenseMode sense        = PowerSenseMode::Analytic; ///< 功率感知方式
    std::vector<double>* history = nullptr; ///< 非空时记录每次迭代的功率，可画收敛曲线
};

/**
 * @brief 功率闭环放大（默认参数）
 * @param input        输入复信号
 * @param target_power 理想输出功率
 * @return 迭代报告
 */
PowerLoopReport amplifyToTargetPower(const ComplexVec& input, double target_power);

/**
 * @brief 功率闭环放大（自定义参数）
 * @param input        输入复信号
 * @param target_power 理想输出功率
 * @param cfg          闭环参数
 * @return 迭代报告
 * @note 标量增益模型用 Analytic，非理想放大器（频响、饱和、AM-PM）用 Measured。
 */
PowerLoopReport amplifyToTargetPower(const ComplexVec& input,
                                     double target_power,
                                     const PowerLoopConfig& cfg);

#endif
