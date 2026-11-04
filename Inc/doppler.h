#ifndef _DOPPLER_H_
#define _DOPPLER_H_

#include <cstddef>
#include <vector>

#include "typedef.h"
#include "filter.h"   ///< MatchFilterMode

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
