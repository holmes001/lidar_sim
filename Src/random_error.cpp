/**
 * @file    random_error.cpp
 * @brief   Gaussian noise injection implementation
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
 * @brief Add complex Gaussian noise with a given total energy (legacy API)
 * @deprecated Noise is normalized by the total energy of the record (see energyZ),
 *             so changing samp changes the actual SNR. Use addGaussianNoiseStd.
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

    std::mt19937 rng(seed);                            ///< Random engine
    std::normal_distribution<double> stdNormal(0.0, 1.0);

    /// Step 1: generate zero-mean standard normal noise and measure its energy
    ComplexVec z(n);
    double energyZ = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        z[i] = stdNormal(rng);
        energyZ += std::abs(z[i]) * std::abs(z[i]) ;
    }

    /// Defensive check: the probability of n consecutive zeros is 0
    if (energyZ <= 0.0)
        return out;

    /// Step 2: scale to the target energy, then add the mean offset
    const double scale = std::sqrt(noiseEnergy / energyZ);
    for (std::size_t i = 0; i < n; ++i)
        out[i] += scale * z[i] + mean;

    return out;
}

/**
 * @brief Add circularly symmetric complex Gaussian noise with a given stddev
 * @param signal Input signal
 * @param stddev Standard deviation of each I/Q component
 * @param mean   Mean, applied to the real part only
 * @param seed   Random seed
 * @return Signal with noise added
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
    /// Independent draws for real and imaginary parts give circular symmetry; mean goes to the real part only
    for (auto& v : out) v += Complex(dist(rng) + mean, dist(rng));
    return out;
}
