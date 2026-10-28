/**
 * @file    filter.h
 * @brief   匹配滤波、FFT 与 range-Doppler 处理
 */
#ifndef FILTER_H
#define FILTER_H

#include <cstddef>
#include <vector>

#include "complex_types.h"

/** @brief 匹配滤波输出模式 */
enum class MatchFilterMode {
    Full,   ///< 长度 N+M-1；峰值位置 = 匹配起点 + M-1
    Same,   ///< 长度 N；输出索引 t 即"信号从 t 开始匹配"的强度
    Valid   ///< 长度 N-M+1（要求 N>=M）；不补零，完全重叠区
};

/**
 * @brief 匹配滤波：y = signal ⊛ h，h[k] = conj(ref[M-1-k])
 * @param signal                输入复信号
 * @param ref                   参考模板，通常取发射波形
 * @param mode                  输出长度与对齐方式
 * @param normalizeByRefEnergy  为 true 时按 ref 总能量归一化，完全匹配时峰值为 1
 * @return 滤波后的复信号；任一输入为空时返回空
 * @note 直接时域卷积，复杂度 O(N·M)。
 */
ComplexVec matchedFilter(const ComplexVec& signal,
                         const ComplexVec& ref,
                         MatchFilterMode mode = MatchFilterMode::Same,
                         bool normalizeByRefEnergy = false);

/**
 * @brief 复数 DFT
 * @param x 输入序列，长度任意（不要求 2 的幂）
 * @return 频谱，长度与输入相同，不归一化
 */
ComplexVec fft(const ComplexVec& x);

/**
 * @brief 复数 IDFT
 * @param x 频谱
 * @return 时域序列，已含 1/N 归一化
 */
ComplexVec ifft(const ComplexVec& x);

/**
 * @brief 频谱搬移，把零频从索引 0 移到中心
 * @param x 输入序列
 * @return 搬移后的序列，y[i] = x[(i + N/2) % N]
 * @note 配合 fft 使用后，索引 N/2 对应 0 Hz，索引递增对应频率递增。
 */
ComplexVec fftShift(const ComplexVec& x);

/**
 * @brief range-Doppler 图的峰值
 *
 * @note 距离取整数列号（时延本身就是整数样点）；多普勒用三点抛物线插值细化，
 *       因此 doppler_hz 不落在频率栅格上。
 */
struct RangeDopplerPeak {
    std::size_t doppler_index = 0;   ///< 多普勒行号（已 fftShift，中心行 = 0 Hz）
    std::size_t range_index   = 0;   ///< 距离列号
    double      doppler_hz    = 0.0; ///< 多普勒频率 [Hz]，正 = 接近
    double      velocity      = 0.0; ///< 径向速度 [m/s]，正 = 接近
    double      range_m       = 0.0; ///< 距离 [m]
    double      power         = 0.0; ///< |RD|²
};

/**
 * @brief 逐脉冲匹配滤波，得到慢时间矩阵
 * @param pulses 各脉冲的回波，pulses[p] 为第 p 个脉冲的快时间记录
 * @param ref    匹配滤波参考模板
 * @param mode   匹配滤波输出模式
 * @return 慢时间矩阵 slow[p][r]，行 = 脉冲序号（慢时间），列 = 距离门
 * @note 各行长度取 pulses[p] 经 matchedFilter 后的实际长度，原则上应一致。
 */
ComplexVec2D matchedFilterFrame(const ComplexVec2D& pulses,
                                const ComplexVec& ref,
                                MatchFilterMode mode = MatchFilterMode::Same);

/**
 * @brief range-Doppler 处理：沿慢时间做 FFT
 * @param slow_time matchedFilterFrame 的输出 slow[p][r]
 * @return range-Doppler 图 rd[k][r]，行 = 多普勒门（已 fftShift，中心行 = 0 Hz），
 *         列 = 距离门，内容为复数值
 * @note 每个距离门独立做 N 点 FFT，一次批量规划；内存 O(N·R)。
 */
ComplexVec2D rangeDopplerMap(const ComplexVec2D& slow_time);

/**
 * @brief 距离轴：R = r · c · dt / 2
 * @param n_range 距离门数
 * @param dt      快时间采样间隔 [s]
 * @return 长度为 n_range 的距离序列 [m]
 */
std::vector<double> rangeAxis(std::size_t n_range, double dt);

/**
 * @brief 多普勒轴（与 fftShift 后的行号对应）
 * @param n_pulse 脉冲数
 * @param pri     脉冲重复间隔 [s]
 * @return 长度为 n_pulse 的多普勒序列 [Hz]，中心行为 0
 * @note 分辨率 1/(N·PRI)，无模糊范围 ±1/(2·PRI)。
 */
std::vector<double> dopplerAxis(std::size_t n_pulse, double pri);

/**
 * @brief 速度轴：v = f_d · λ / 2
 * @param n_pulse    脉冲数
 * @param pri        脉冲重复间隔 [s]
 * @param wavelength 中心波长 [m]
 * @return 长度为 n_pulse 的径向速度序列 [m/s]
 */
std::vector<double> velocityAxis(std::size_t n_pulse, double pri, double wavelength);

/**
 * @brief 在 range-Doppler 图中搜索峰值并换算物理量
 * @param rd         rangeDopplerMap 的输出
 * @param pri        脉冲重复间隔 [s]
 * @param dt         快时间采样间隔 [s]
 * @param wavelength 中心波长 [m]
 * @return 峰值信息；rd 为空时返回全零
 * @note 多普勒用三点抛物线插值细化，峰值贴边时退化为整数行号。
 */
RangeDopplerPeak findRangeDopplerPeak(const ComplexVec2D& rd,
                                      double pri,
                                      double dt,
                                      double wavelength);

#endif
