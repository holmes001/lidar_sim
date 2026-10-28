#ifndef RANDOM_ERROR_H
#define RANDOM_ERROR_H

#include <complex>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <string>

/* 按"整段信号的总能量"归一化：改变采样点数 samp 会改变实际 SNR。
 * 新代码请用 addGaussianNoiseStd（按标准差注入，不做能量归一化）。 */
std::vector<std::complex<double>> addGaussianNoise(const std::vector<std::complex<double>>& signal,
                                     double noiseEnergy,
                                     double mean = 0.0,
                                     unsigned int seed = std::random_device{}());

std::vector<std::complex<double>> addGaussianNoiseStd(const std::vector<std::complex<double>>& signal,
                                        double stddev,
                                        double mean = 0.0,
                                        unsigned int seed = std::random_device{}());

#endif
