#include "random_error.h"

#include <complex>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <string>

/* 注意：本函数按"整段信号的总能量"归一化（见下面的 energyZ），
 * 因此改变采样点数 samp 就会改变实际 SNR。新代码请用 addGaussianNoiseStd。 */
std::vector<std::complex<double>> addGaussianNoise(const std::vector<std::complex<double>>& signal,
                                     double noiseEnergy,
                                     double mean,
                                     unsigned int seed)
{
    if (noiseEnergy < 0.0)
        throw std::invalid_argument("noiseEnergy must be >= 0");

    std::vector<std::complex<double>> out(signal);
    const std::size_t n = out.size();
    if (n == 0 || noiseEnergy == 0.0)
        return out;

    // 随机数引擎 + 标准正态分布
    std::mt19937 rng(seed);
    std::normal_distribution<double> stdNormal(0.0, 1.0);

    // 第一步：生成零均值标准正态噪声，并统计其能量
    std::vector<std::complex<double>> z(n);
    double energyZ = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        z[i] = stdNormal(rng);
        energyZ += std::abs(z[i]) * std::abs(z[i]) ;
    }

    // 极端保护（n 个连续随机变量同时为 0 的概率为 0，仅防御性处理）
    if (energyZ <= 0.0)
        return out;

    // 第二步：缩放到目标能量，再叠加均值偏移
    const double scale = std::sqrt(noiseEnergy / energyZ);
    for (std::size_t i = 0; i < n; ++i)
        out[i] += scale * z[i] + mean;

    return out;
}

std::vector<std::complex<double>> addGaussianNoiseStd(const std::vector<std::complex<double>>& signal,
                                        double stddev,
                                        double mean,
                                        unsigned int seed)
{
    if (stddev < 0.0)
        throw std::invalid_argument("stddev must be >= 0");

    std::vector<std::complex<double>> out(signal);
    if (out.empty() || (stddev == 0.0 && mean == 0.0))
        return out;

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, stddev);
    // 实部与虚部各自独立抽样，才是圆对称复高斯；mean 只加在实部。
    for (auto& v : out) v += std::complex<double>(dist(rng) + mean, dist(rng));
    return out;
}
