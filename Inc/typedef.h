/**
 * @file    typedef.h
 * @brief   全工程统一的类型别名与物理常量
 *
 * @note 需要复数的文件只包含本头文件，不要在别处重复定义 Complex / ComplexVec，
 *       以免出现多套不一致的类型。
 * @note kpi / c 是全工程唯一的常量来源，不要在别处重复定义。
 * @note 本头文件**不依赖任何项目内其他头文件、也不拉入 OpenCV**；
 *       需要绘图（pl::Style / cv::Mat）的文件请自行包含 plot.h。
 */
#ifndef _TYPEDEF_H
#define _TYPEDEF_H

#include <complex>
#include <vector>

using Complex    = std::complex<double>;      ///< 单点复数
using ComplexVec = std::vector<Complex>;      ///< 一维复信号
using ComplexVec2D = std::vector<ComplexVec>; ///< 二维复信号（按行存各脉冲）

using DoubleVec = std::vector<double>;        ///< 一维实信号

/* ---- 物理与数学常量 ---- */

constexpr double kpi = 3.14159265358979323846;   ///< 圆周率
const double c = 3.0e8;                          ///< 光速 [m/s]

#endif
