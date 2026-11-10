/**
 * @file    random_error.h
 * @brief   Gaussian noise injection
 */
#ifndef RANDOM_ERROR_H
#define RANDOM_ERROR_H

#include "typedef.h"

#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <string>

/**
 * @brief Add complex Gaussian noise with a given total energy (legacy API)
 * @param signal      Input signal
 * @param noiseEnergy Target total noise energy sum(|n|^2)
 * @param mean        Mean, applied to the real part only
 * @param seed        Random seed
 * @return Signal with noise added
 * @deprecated Noise is normalized by the total energy of the record, so changing
 *             samp changes the actual SNR. Use addGaussianNoiseStd instead.
 */
ComplexVec addGaussianNoise(const ComplexVec& signal,
                                     double noiseEnergy,
                                     double mean = 0.0,
                                     unsigned int seed = std::random_device{}());

/**
 * @brief Add circularly symmetric complex Gaussian noise with a given stddev
 * @param signal Input signal
 * @param stddev Standard deviation of each I/Q component
 * @param mean   Mean, applied to the real part only
 * @param seed   Random seed
 * @return Signal with noise added
 * @note Real and imaginary parts are sampled independently, so the noise power
 *       does not depend on the number of samples.
 */
ComplexVec addGaussianNoiseStd(const ComplexVec& signal,
                                        double stddev,
                                        double mean = 0.0,
                                        unsigned int seed = std::random_device{}());

#endif
