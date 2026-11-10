/**
 * @file    typedef.h
 * @brief   Project-wide type aliases and physical constants
 *
 * @note Files that need complex numbers include only this header. Never redefine
 *       Complex / ComplexVec elsewhere, or the project ends up with multiple
 *       inconsistent types.
 * @note kpi / c are the single source of truth; do not redefine them.
 * @note This header depends on no other project header and does **not** pull in
 *       OpenCV. Files that need plotting (pl::Style / cv::Mat) include plot.h.
 */
#ifndef _TYPEDEF_H
#define _TYPEDEF_H

#include <complex>
#include <vector>

using Complex    = std::complex<double>;      ///< Single complex sample
using ComplexVec = std::vector<Complex>;      ///< 1-D complex signal
using ComplexVec2D = std::vector<ComplexVec>; ///< 2-D complex signal (one row per pulse)

using DoubleVec = std::vector<double>;        ///< 1-D real signal

/* ---- Physical and mathematical constants ---- */

constexpr double kpi = 3.14159265358979323846;   ///< Pi
const double c = 3.0e8;                          ///< Speed of light [m/s]

#endif
