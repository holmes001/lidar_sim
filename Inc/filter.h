/**
 * @file    filter.h
 * @brief   匹配滤波
 */
#ifndef FILTER_H
#define FILTER_H

#include <cstddef>
#include <vector>

#include "typedef.h"

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

#endif
