/**
 * @file    filter.h
 * @brief   Matched filtering
 */
#ifndef FILTER_H
#define FILTER_H

#include <cstddef>
#include <vector>

#include "typedef.h"

/** @brief Matched filter output mode */
enum class MatchFilterMode {
    Full,   ///< Length N+M-1; peak position = match start + M-1
    Same,   ///< Length N; index t is the score of "matching a signal from t"
    Valid   ///< Length N-M+1 (requires N>=M); no zero padding, overlap only
};

/**
 * @brief Matched filter: y = signal conv h, with h[k] = conj(ref[M-1-k])
 * @param signal                Input complex signal
 * @param ref                   Reference template, usually the transmit waveform
 * @param mode                  Output length and alignment
 * @param normalizeByRefEnergy  If true, scale by the total energy of ref so a
 *                              perfect match peaks at 1
 * @return Filtered complex signal; empty if either input is empty
 * @note Direct time-domain convolution, complexity O(N*M).
 */
ComplexVec matchedFilter(const ComplexVec& signal,
                         const ComplexVec& ref,
                         MatchFilterMode mode = MatchFilterMode::Same,
                         bool normalizeByRefEnergy = false);

#endif
