/**
 * @file    random_error.h
 * @brief   高斯噪声注入
 */
#ifndef RANDOM_ERROR_H
#define RANDOM_ERROR_H

#include "complex_types.h"

#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <string>

/**
 * @brief 按总能量注入复高斯噪声（旧接口）
 * @param signal      输入信号
 * @param noiseEnergy 目标噪声总能量 Σ|n|²
 * @param mean        均值，只加在实部
 * @param seed        随机种子
 * @return 叠加噪声后的信号
 * @deprecated 噪声按整段信号总能量归一化，改变采样点数 samp 会改变实际 SNR。
 *             新代码请用 addGaussianNoiseStd。
 */
ComplexVec addGaussianNoise(const ComplexVec& signal,
                                     double noiseEnergy,
                                     double mean = 0.0,
                                     unsigned int seed = std::random_device{}());

/**
 * @brief 按标准差注入圆对称复高斯噪声
 * @param signal 输入信号
 * @param stddev 每个 I/Q 分量的标准差
 * @param mean   均值，只加在实部
 * @param seed   随机种子
 * @return 叠加噪声后的信号
 * @note 实部与虚部各自独立抽样，噪声功率与采样点数无关。
 */
ComplexVec addGaussianNoiseStd(const ComplexVec& signal,
                                        double stddev,
                                        double mean = 0.0,
                                        unsigned int seed = std::random_device{}());

#endif
