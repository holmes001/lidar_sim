/**
 * @file    app.h
 * @brief   通用小工具：FFT、取样、坐标轴、存图
 *
 * @note 这里只放与物理模型无关的通用函数：
 *       雷达方程/波形/回波在 transport.h，匹配滤波在 filter.h，
 *       多普勒处理在 doppler.h，功率与闭环在 power.h。
 */
#ifndef _APP_H_
#define _APP_H_

#include <cstddef>
#include <string>
#include <vector>

#include "typedef.h"      ///< Complex / ComplexVec
#include "plot.h"         ///< cv::Mat / pl::Style

namespace pl = lidar::plot;

/* ---- 复数变换 ---- */

/** @brief 复数 DFT（长度任意，不归一化） */
ComplexVec fft(const ComplexVec& x);

/** @brief 复数 IDFT（含 1/N 归一化） */
ComplexVec ifft(const ComplexVec& x);

/** @brief 频谱搬移：把零频从索引 0 移到中心 */
ComplexVec fftShift(const ComplexVec& x);

/* ---- 取样 ---- */

std::vector<double> vector_real(const ComplexVec &u);
std::vector<double> vector_abs(const ComplexVec &u);

/* ---- 坐标轴与窗 ---- */

std::vector<double> time_axis(std::size_t n, double dt);
void crop_window(const std::vector<double> &t_us, const std::vector<double> &y,
                 double t0, double t1,
                 std::vector<double> &t_out, std::vector<double> &y_out);

/* ---- 存图 ---- */

bool save_figure(const cv::Mat &fig, const std::string &file_path);
bool save_picture(const cv::Mat &a, const pl::Style &st, const std::string &file_path);

#endif /* _APP_H_ */
