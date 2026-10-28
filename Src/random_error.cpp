/**
 * @file    random_error.cpp
 * @brief   高斯噪声注入实现
 */
#include "random_error.h"

#include <complex>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <string>

/**
 * @brief 按总能量注入复高斯噪声（旧接口）
 * @deprecated 噪声按整段信号总能量归一化（见 energyZ），改变采样点数 samp
 *             会改变实际 SNR。新代码请用 addGaussianNoiseStd。
 */
ComplexVec addGaussianNoise(const ComplexVec& signal,
                                     double noiseEnergy,
                                     double mean,
                                     unsigned int seed)
{
    if (noiseEnergy < 0.0)
        throw std::invalid_argument("noiseEnergy must be >= 0");

    ComplexVec out(signal);
    const std::size_t n = out.size();
    if (n == 0 || noiseEnergy == 0.0)
        return out;

    std::mt19937 rng(seed);                            ///< 随机数引擎
    std::normal_distribution<double> stdNormal(0.0, 1.0);

    /// 第一步：生成零均值标准正态噪声，并统计其能量
    ComplexVec z(n);
    double energyZ = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        z[i] = stdNormal(rng);
        energyZ += std::abs(z[i]) * std::abs(z[i]) ;
    }

    /// 防御性处理：n 个连续随机变量同时为 0 的概率为 0
    if (energyZ <= 0.0)
        return out;

    /// 第二步：缩放到目标能量，再叠加均值偏移
    const double scale = std::sqrt(noiseEnergy / energyZ);
    for (std::size_t i = 0; i < n; ++i)
        out[i] += scale * z[i] + mean;

    return out;
}

/**
 * @brief 按标准差注入圆对称复高斯噪声
 * @param signal 输入信号
 * @param stddev 每个 I/Q 分量的标准差
 * @param mean   均值，只加在实部
 * @param seed   随机种子
 * @return 叠加噪声后的信号
 */
ComplexVec addGaussianNoiseStd(const ComplexVec& signal,
                                        double stddev,
                                        double mean,
                                        unsigned int seed)
{
    if (stddev < 0.0)
        throw std::invalid_argument("stddev must be >= 0");

    ComplexVec out(signal);
    if (out.empty() || (stddev == 0.0 && mean == 0.0))
        return out;

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, stddev);
    /// 实部与虚部各自独立抽样，才是圆对称复高斯；mean 只加在实部
    for (auto& v : out) v += Complex(dist(rng) + mean, dist(rng));
    return out;
}
