/**
 * @file    filter.cpp
 * @brief   匹配滤波实现
 */
#include "filter.h"

#include <algorithm>
#include <cstddef>

/**
 * @brief 匹配滤波：y = signal ⊛ h，h[k] = conj(ref[M-1-k])
 * @param signal                输入复信号
 * @param ref                   参考模板
 * @param mode                  输出长度与对齐方式
 * @param normalizeByRefEnergy  为 true 时按 ref 总能量归一化
 * @return 滤波后的复信号；任一输入为空时返回空
 */
ComplexVec matchedFilter(const ComplexVec& signal,
                         const ComplexVec& ref,
                         MatchFilterMode mode,
                         bool normalizeByRefEnergy)
{
    const std::size_t N = signal.size();
    const std::size_t M = ref.size();
    if (N == 0 || M == 0) return {};

    /// 直接时域卷积：y_full[n] = Σ_k signal[k]·conj(ref[M-1-(n-k)])
    ComplexVec full(N + M - 1, Complex(0.0, 0.0));
    for (std::size_t n = 0; n < N; ++n) {
        const Complex xn = signal[n];
        if (xn == Complex(0.0, 0.0)) continue;          ///< 稀疏信号优化
        for (std::size_t k = 0; k < M; ++k)
            full[n + k] += xn * std::conj(ref[M - 1 - k]);
    }

    /// 可选归一化：按参考信号总能量缩放，完全匹配时峰值 = 1
    if (normalizeByRefEnergy) {
        double e = 0.0;
        for (const auto& c : ref) e += std::norm(c);
        if (e > 0.0) {
            const double inv = 1.0 / e;
            for (auto& c : full) c *= inv;
        }
    }

    switch (mode) {
    case MatchFilterMode::Full:
        return full;

    case MatchFilterMode::Same: {
        /// 取 full[M-1 : M-1+N]，索引 t 对应信号从 t 起与 ref 匹配
        ComplexVec out(N);
        std::copy(full.begin() + (M - 1),
                  full.begin() + (M - 1 + N),
                  out.begin());
        return out;
    }

    case MatchFilterMode::Valid: {
        if (N < M) return {};                          ///< 无完全重叠区
        const std::size_t L = N - M + 1;
        ComplexVec out(L);
        std::copy(full.begin() + (M - 1),
                  full.begin() + (M - 1 + L),
                  out.begin());
        return out;
    }
    }
    return {};
}
