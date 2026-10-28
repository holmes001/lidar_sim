#ifndef FILTER_H
#define FILTER_H

#include <complex>
#include <cstddef>
#include <vector>

using Complex    = std::complex<double>;
using ComplexVec = std::vector<Complex>;

enum class MatchFilterMode {
    Full,   // 长度 N+M-1，完整卷积；峰值位置 = 匹配起点 + M-1
    Same,   // 长度 N（与输入等长）；输出索引 t 即"信号从 t 开始匹配"的强度
    Valid   // 长度 N-M+1（要求 N>=M）；无 zero-padding，完全重叠区
};

// ---- 匹配滤波 ----
//   signal  输入复信号
//   ref     参考(模板)信号，通常取发射脉冲本身
//   mode    输出长度/对齐方式
//   normalizeByRefEnergy  为 true 时用 ref 总能量归一化（完全匹配时峰值 = 1）
ComplexVec matchedFilter(const ComplexVec& signal,
                         const ComplexVec& ref,
                         MatchFilterMode mode = MatchFilterMode::Same,
                         bool normalizeByRefEnergy = false);

// ---- 信号功率测量 ----
double      signalEnergy(const ComplexVec& x);                       // 总能量 Σ|x|^2
double      peakPower(const ComplexVec& x);                          // 峰值功率 max|x|^2
std::size_t peakPowerIndex(const ComplexVec& x);                     // 峰值位置
double      powerToDB(double power, double ref = 1.0);               // 10*log10(P/ref)
double      measureSignalPower(const ComplexVec& x,
                               bool in_db = false,
                               double ref = 1.0);                    // 平均功率

// -----------------------------------------------------------------
// 离散 PI 控制器（位置式）
//   u[k] = kp * e[k] + Σ_i (ki * e[i])
// e 为归一化误差(无量纲)，u 为归一化控制量(无量纲)，
// 因此 kp、ki 与信号量纲无关，便于移植。
// -----------------------------------------------------------------
struct PIController {
    double kp       = 1.0;      // 比例增益
    double ki       = 0.2;      // 积分增益(每步累积量)
    double out_min  = -0.99;    // 控制量下限，保证 1+u > 0
    double out_max  = 1.0e3;    // 控制量上限
    double integ_gate = 1.0;    // 积分分离阈值：|e| > 它时只走比例
                                // (防止远距离目标把积分冲爆而发散)
    double integral = 0.0;      // 积分累积项

    // 单步更新：输入归一化误差，返回限幅后的控制量
    double update(double error);
    void   reset() { integral = 0.0; }
};

// 功率闭环放大的迭代报告
struct PowerLoopReport {
    ComplexVec output;          // 闭环输出信号 y = g * x
    double     gain       = 1.0;// 最终(总)增益
    double     power      = 0.0;// 输出实际功率
    double     abs_error  = 0.0;// 绝对误差 = 理想功率 - 实际功率
    double     rel_error  = 0.0;// 相对误差 = |abs_error| / 理想功率
    int        iterations = 0;  // 实际迭代次数
    bool       converged  = false; // 是否满足收敛阈值
};

// 功率感知方式
enum class PowerSenseMode {
    Analytic,   // 解析：y = g*x => P = g^2 * P_in，每轮 O(1)，不重建信号
    Measured    // 实测：每轮重建 y 并遍历求功率，O(N)；
                // 频域响应/饱和/非线性模型必须用这个
};

// 功率闭环放大器的全部可调参数（都有默认值，按需覆盖即可）
struct PowerLoopConfig {
    double         kp           = 1.0;    // PI 比例增益
    double         ki           = 0.2;    // PI 积分增益
    double         integ_gate   = 1.0;    // 积分分离阈值：|e_rel| 超过它时只走比例
    double         tol_rel      = 1.0e-3; // 收敛阈值：相对误差 |err|/target 小于它即停
    int            max_iter     = 100;    // 最大迭代次数
    double         initial_gain = 1.0;    // 初始增益 g0
    double         max_gain     = 1.0e6;  // 增益上限（防止发散）
    PowerSenseMode sense        = PowerSenseMode::Analytic; // 功率感知方式
    std::vector<double>* history = nullptr; // 非空时记录每次迭代的功率(可画收敛曲线)
};

// 功率闭环放大器：默认参数版本
PowerLoopReport amplifyToTargetPower(const ComplexVec& input, double target_power);

// 功率闭环放大器：自定义参数版本
//   输入复信号 + 理想功率，迭代调节增益 g 使输出平均功率趋于理想功率。
//   sense 标量增益模型用 Analytic(默认)，非理想放大器用 Measured
PowerLoopReport amplifyToTargetPower(const ComplexVec& input,
                                     double target_power,
                                     const PowerLoopConfig& cfg);

#endif
