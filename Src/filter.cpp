/**
 * @file    filter.cpp
 * @brief   Matched filter implementation
 */
#include "filter.h"

#include <algorithm>
#include <cstddef>

/**
 * @brief Matched filter: y = signal conv h, with h[k] = conj(ref[M-1-k])
 * @param signal                Input complex signal
 * @param ref                   Reference template
 * @param mode                  Output length and alignment
 * @param normalizeByRefEnergy  If true, normalize by the total energy of ref
 * @return Filtered complex signal; empty if either input is empty
 */
ComplexVec matchedFilter(const ComplexVec& signal,
                         const ComplexVec& ref,
                         MatchFilterMode mode,
                         bool normalizeByRefEnergy)
{
    const std::size_t N = signal.size();
    const std::size_t M = ref.size();
    if (N == 0 || M == 0) return {};

    /// Direct time-domain convolution: y_full[n] = sum_k signal[k]*conj(ref[M-1-(n-k)])
    ComplexVec full(N + M - 1, Complex(0.0, 0.0));
    for (std::size_t n = 0; n < N; ++n) {
        const Complex xn = signal[n];
        if (xn == Complex(0.0, 0.0)) continue;          ///< Sparse-signal fast path
        for (std::size_t k = 0; k < M; ++k)
            full[n + k] += xn * std::conj(ref[M - 1 - k]);
    }

    /// Optional normalization: scale by the reference energy so a perfect match peaks at 1
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
        /// Take full[M-1 : M-1+N]; index t means the signal is matched starting at t
        ComplexVec out(N);
        std::copy(full.begin() + (M - 1),
                  full.begin() + (M - 1 + N),
                  out.begin());
        return out;
    }

    case MatchFilterMode::Valid: {
        if (N < M) return {};                          ///< No full-overlap region
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
