/**
 * @file    complex_types.h
 * @brief   全工程统一的复数与复信号类型别名
 *
 * @note 需要复数的文件只包含本头文件，不要在别处重复定义 Complex / ComplexVec，
 *       以免出现多套不一致的类型。
 */
#ifndef COMPLEX_TYPES_H
#define COMPLEX_TYPES_H

#include <complex>
#include <vector>

using Complex    = std::complex<double>;      ///< 单点复数
using ComplexVec = std::vector<Complex>;      ///< 一维复信号
using ComplexVec2D = std::vector<ComplexVec>; ///< 二维复信号（按行存各脉冲）

using DoubleVec = std::vector<double>;        ///< 一维实信号

#endif
