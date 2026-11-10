#ifndef _DOPPLER_H_
#define _DOPPLER_H_

#include <cstddef>
#include <vector>

#include "typedef.h"
#include "filter.h"   ///< MatchFilterMode

struct RangeDopplerPeak {
    std::size_t doppler_index = 0;   ///< Doppler row (fftShift'ed, center row = 0 Hz)
    std::size_t range_index   = 0;   ///< Range column
    double      doppler_hz    = 0.0; ///< Doppler frequency [Hz], positive = approaching
    double      velocity      = 0.0; ///< Radial velocity [m/s], positive = approaching
    double      range_m       = 0.0; ///< Range [m]
    double      power         = 0.0; ///< |RD|^2
};

/**
 * @brief Per-pulse matched filtering, producing the slow-time matrix
 * @param pulses Echoes per pulse; pulses[p] is the fast-time record of pulse p
 * @param ref    Matched filter reference template
 * @param mode   Matched filter output mode
 * @return Slow-time matrix slow[p][r], rows = pulse index (slow time), cols = range bin
 * @note Row length is the actual length returned by matchedFilter, normally equal.
 */
ComplexVec2D matchedFilterFrame(const ComplexVec2D& pulses,
                                const ComplexVec& ref,
                                MatchFilterMode mode = MatchFilterMode::Same);

/**
 * @brief Range-Doppler processing: FFT along slow time
 * @param slow_time Output of matchedFilterFrame, slow[p][r]
 * @return Range-Doppler map rd[k][r], rows = Doppler bin (fftShift'ed, center row =
 *         0 Hz), cols = range bin, values are complex
 * @note One N-point FFT per range bin with a single batched plan; memory O(N*R).
 */
ComplexVec2D rangeDopplerMap(const ComplexVec2D& slow_time);

/**
 * @brief Range axis: R = r * c * dt / 2
 * @param n_range Number of range bins
 * @param dt      Fast-time sampling interval [s]
 * @return Range sequence of length n_range [m]
 */
std::vector<double> rangeAxis(std::size_t n_range, double dt);

/**
 * @brief Doppler axis (matching the row order after fftShift)
 * @param n_pulse Number of pulses
 * @param pri     Pulse repetition interval [s]
 * @return Doppler sequence of length n_pulse [Hz]; center row is 0
 * @note Resolution 1/(N*PRI), unambiguous range +/-1/(2*PRI).
 */
std::vector<double> dopplerAxis(std::size_t n_pulse, double pri);

/**
 * @brief Velocity axis: v = f_d * lambda / 2
 * @param n_pulse    Number of pulses
 * @param pri        Pulse repetition interval [s]
 * @param wavelength Center wavelength [m]
 * @return Radial velocity sequence of length n_pulse [m/s]
 */
std::vector<double> velocityAxis(std::size_t n_pulse, double pri, double wavelength);

/**
 * @brief Find the peak in the range-Doppler map and convert it to physical units
 * @param rd         Output of rangeDopplerMap
 * @param pri        Pulse repetition interval [s]
 * @param dt         Fast-time sampling interval [s]
 * @param wavelength Center wavelength [m]
 * @return Peak information; all zeros when rd is empty
 * @note Doppler is refined by three-point parabolic interpolation; when the peak
 *       sits on the first or last row it falls back to the integer row index.
 */
RangeDopplerPeak findRangeDopplerPeak(const ComplexVec2D& rd,
                                      double pri,
                                      double dt,
                                      double wavelength);

#endif
