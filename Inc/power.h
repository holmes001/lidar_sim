/**
 * @file    power.h
 * @brief   Signal power measurement and closed-loop power amplification (discrete PI)
 */
#ifndef POWER_H
#define POWER_H

#include <cstddef>

#include "typedef.h"

/**
 * @name Signal power measurement
 * @{
 */

/** @brief Total energy E = sum |x[n]|^2 */
double      signalEnergy(const ComplexVec& x);

/** @brief Peak power max |x[n]|^2 */
double      peakPower(const ComplexVec& x);

/** @brief Sample index of the peak power; returns 0 for an empty signal */
std::size_t peakPowerIndex(const ComplexVec& x);

/**
 * @brief Power to dB: 10*log10(P / ref)
 * @param power Power, must be positive
 * @param ref   Reference power, default 1.0
 * @return dB value; -inf when power or ref is not positive
 */
double      powerToDB(double power, double ref = 1.0);

/**
 * @brief Average power P = (1/N)*sum |x[n]|^2
 * @param x     Input complex signal
 * @param in_db If true, return the value in dB
 * @param ref   Reference power for dB normalization
 * @return Average power; 0 for an empty signal
 * @note Suited to whole-record signals such as noise or echoes. Use peakPower when
 *       only the matched filter peak matters.
 */
double      measureSignalPower(const ComplexVec& x,
                               bool in_db = false,
                               double ref = 1.0);

/** @} */

/**
 * @brief Discrete PI controller (position form, with integral separation and anti-windup)
 *
 * u[k] = kp*e[k] + sum_i ki*e[i]. Both e (normalized error) and u (normalized
 * control) are dimensionless, so kp/ki are independent of signal units.
 *
 * @warning integral is state: call reset() before switching to a new plant or
 *          restarting the loop, otherwise the accumulated integral carries over.
 */
struct PIController {
    double kp       = 1.0;      ///< Proportional gain
    double ki       = 0.2;      ///< Integral gain (per-step increment)
    double out_min  = -0.99;    ///< Lower control limit, keeps 1+u > 0
    double out_max  = 1.0e3;    ///< Upper control limit
    double integ_gate = 1.0;    ///< Integral separation threshold: |e| above it is P-only
    double integral = 0.0;      ///< Accumulated integral term

    /// Single-step update: takes the normalized error, returns the clamped control
    double update(double error);
    /// Clear the integral term
    void   reset() { integral = 0.0; }
};

/**
 * @brief Iteration report of the closed-loop power amplifier
 *
 * @note Fields are refreshed on every iteration and hold the final values after
 *       convergence.
 */
struct PowerLoopReport {
    ComplexVec output;             ///< Closed-loop output y = g*x
    double     gain       = 1.0;   ///< Final (total) gain
    double     power      = 0.0;   ///< Measured output power
    double     abs_error  = 0.0;   ///< Absolute error = target power - actual power
    double     rel_error  = 0.0;   ///< Relative error = |abs_error| / target power
    int        iterations = 0;     ///< Actual number of iterations
    bool       converged  = false; ///< Whether the convergence threshold was met
};

/** @brief How output power is sensed */
enum class PowerSenseMode {
    Analytic,   ///< Analytic: P = g^2*P_in, O(1) per iteration, no signal rebuild
    Measured    ///< Measured: rebuild y each iteration and measure it, O(N); required
                ///< for non-scalar models
};

/**
 * @brief All tunable parameters of the closed-loop power amplifier
 *
 * @warning history points to an external container: it must stay alive while this
 *          structure is in use and must not be destroyed early.
 */
struct PowerLoopConfig {
    double         kp           = 1.0;    ///< PI proportional gain
    double         ki           = 0.2;    ///< PI integral gain
    double         integ_gate   = 1.0;    ///< Integral separation threshold: |e_rel| above it is P-only
    double         tol_rel      = 1.0e-3; ///< Convergence threshold: stop when relative error is below it
    int            max_iter     = 100;    ///< Maximum number of iterations
    double         initial_gain = 1.0;    ///< Initial gain g0
    double         max_gain     = 1.0e6;  ///< Gain ceiling, prevents divergence
    PowerSenseMode sense        = PowerSenseMode::Analytic; ///< Power sensing mode
    std::vector<double>* history = nullptr; ///< If non-null, records the power of each iteration for plotting
};

/**
 * @brief Closed-loop power amplification (default parameters)
 * @param input        Input complex signal
 * @param target_power Target output power
 * @return Iteration report
 */
PowerLoopReport amplifyToTargetPower(const ComplexVec& input, double target_power);

/**
 * @brief Closed-loop power amplification (custom parameters)
 * @param input        Input complex signal
 * @param target_power Target output power
 * @param cfg          Loop parameters
 * @return Iteration report
 * @note Use Analytic for scalar gain models and Measured for non-ideal amplifiers
 *       (frequency response, saturation, AM-PM).
 */
PowerLoopReport amplifyToTargetPower(const ComplexVec& input,
                                     double target_power,
                                     const PowerLoopConfig& cfg);

#endif
