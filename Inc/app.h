/**
 * @file    app.h
 * @brief   Generic helpers: FFT, sampling, axes, figure saving
 *
 * @note Only model-agnostic helpers live here: radar equation / waveforms / echoes
 *       in transport.h, matched filtering in filter.h, Doppler processing in
 *       doppler.h, power and closed loop in power.h.
 */
#ifndef _APP_H_
#define _APP_H_

#include <cstddef>
#include <string>
#include <vector>

#include "typedef.h"      ///< Complex / ComplexVec
#include "plot.h"         ///< cv::Mat / pl::Style

namespace pl = lidar::plot;

/* ---- Complex transforms ---- */

/** @brief Complex DFT (any length, not normalized) */
ComplexVec fft(const ComplexVec& x);

/** @brief Complex IDFT (with 1/N normalization) */
ComplexVec ifft(const ComplexVec& x);

/** @brief Spectrum shift: move zero frequency from index 0 to the center */
ComplexVec fftShift(const ComplexVec& x);

/* ---- Sampling ---- */

std::vector<double> vector_real(const ComplexVec &u);
std::vector<double> vector_abs(const ComplexVec &u);

/* ---- Axes and windows ---- */

std::vector<double> time_axis(std::size_t n, double dt);
void crop_window(const std::vector<double> &t_us, const std::vector<double> &y,
                 double t0, double t1,
                 std::vector<double> &t_out, std::vector<double> &y_out);

/* ---- Figure output ---- */

bool save_figure(const cv::Mat &fig, const std::string &file_path);
bool save_picture(const cv::Mat &a, const pl::Style &st, const std::string &file_path);

#endif /* _APP_H_ */
