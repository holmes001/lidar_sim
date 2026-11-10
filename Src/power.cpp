/**
 * @file    power.cpp
 * @brief   Signal power measurement and closed-loop power amplification
 */
#include "power.h"

#include <algorithm>
#include <cmath>
#include <limits>

/**
 * @name Signal power measurement
 * @{
 */

/// Total energy: E = sum |x[n]|^2
double signalEnergy(const ComplexVec& x)
{
    double e = 0.0;
    for (const auto& c : x) e += std::norm(c);
    return e;
}

/// Peak power: P_peak = max |x[n]|^2
double peakPower(const ComplexVec& x)
{
    double p = 0.0;
    for (const auto& c : x) p = std::max(p, std::norm(c));
    return p;
}

/// Index of the peak power; returns 0 for an empty signal
std::size_t peakPowerIndex(const ComplexVec& x)
{
    std::size_t idx = 0;
    double best = -1.0;
    for (std::size_t n = 0; n < x.size(); ++n) {
        const double p = std::norm(x[n]);
        if (p > best) { best = p; idx = n; }
    }
    return idx;
}

/// Power to dB: 10*log10(P / ref); returns -inf when P <= 0
double powerToDB(double power, double ref)
{
    if (power <= 0.0 || ref <= 0.0)
        return -std::numeric_limits<double>::infinity();
    return 10.0 * std::log10(power / ref);
}

/**
 * @brief Average power P = (1/N)*sum |x[n]|^2
 * @param x     Input complex signal
 * @param in_db If true, return the value in dB (relative to 1.0, i.e. dBW/dBFS)
 * @param ref   Reference power for dB normalization
 * @return Average power; 0 for an empty signal
 * @note Average power = total energy / sample count, suited to whole-record signals
 *       such as noise or echoes. Use peakPower when only the matched filter peak
 *       matters.
 */
double measureSignalPower(const ComplexVec& x,
                          bool in_db,
                          double ref)
{
    if (x.empty()) return 0.0;

    const double p = signalEnergy(x) / static_cast<double>(x.size());
    return in_db ? powerToDB(p, ref) : p;
}

/** @} */

/**
 * @brief Single-step update of the discrete PI controller
 * @param error Normalized error (dimensionless)
 * @return Clamped control value
 *
 * @note Three anti-windup rules:
 *       1) Integral separation: when |e| exceeds integ_gate, skip the integral and
 *          use proportional action only;
 *       2) Freeze the integral once the output is saturated high while the error is
 *          still positive;
 *       3) Freeze the integral once the output is saturated low while the error is
 *          still negative.
 */
double PIController::update(double error)
{
    double u = kp * error + integral;                  ///< Position-form PI
    u = std::clamp(u, out_min, out_max);

    const bool sat_high  = (u >= out_max) && (error > 0.0);
    const bool sat_low   = (u <= out_min) && (error < 0.0);
    const bool gate_open = std::abs(error) <= integ_gate;   ///< Integral separation
    if (gate_open && !sat_high && !sat_low)
        integral += ki * error;                        ///< Integrate only while unsaturated

    return u;
}

/// Closed-loop power amplification: default parameters
PowerLoopReport amplifyToTargetPower(const ComplexVec& input, double target_power)
{
    return amplifyToTargetPower(input, target_power, PowerLoopConfig{});
}

/**
 * @brief Closed-loop power amplification: adjust the gain g iteratively until the
 *        average output power reaches the target
 * @param input        Input complex signal
 * @param target_power Target output power
 * @param cfg          Loop parameters
 * @return Iteration report
 *
 * @note The plant is y[n] = g*x[n] with average power P = g^2*P_in, which is
 *       nonlinear in g, so the error is normalized on power and the correction is
 *       square-rooted before being applied to the gain:
 *       e = P_target - P_meas, e_rel = e/P_meas, u = PI(e_rel),
 *       P_des = P_meas*(1+u), g <- g*sqrt(1+u).
 *       With kp = 1, P_des = P_target in a single step; the integral removes the
 *       residual.
 * @note In Analytic mode P_in is computed once outside the loop, each iteration is
 *       O(1), and the output is rebuilt only at the end; Measured mode rebuilds y and
 *       measures it every iteration (O(N)), which non-scalar models require.
 */
PowerLoopReport amplifyToTargetPower(const ComplexVec& input,
                                     double target_power,
                                     const PowerLoopConfig& cfg)
{
    const double kp           = cfg.kp;
    const double ki           = cfg.ki;
    const int    max_iter     = cfg.max_iter;
    const double initial_gain = cfg.initial_gain;
    const double max_gain     = cfg.max_gain;
    std::vector<double>* const history = cfg.history;   ///< Points to an external container, may be null
    PowerLoopReport rep;
    rep.output = input;                    ///< Pass through if it never converges / on error
    rep.gain   = initial_gain;

    if (input.empty() || target_power <= 0.0 || max_iter <= 0 || max_gain <= 0.0)
        return rep;

    /// Input power is computed once: for a scalar gain P_out = g^2*P_in
    const double p_in = measureSignalPower(input);

    /// An all-zero input cannot be amplified to any power
    if (p_in <= 0.0) {
        rep.power     = 0.0;
        rep.abs_error = target_power;
        rep.rel_error = 1.0;               ///< Actual power is 0, so the relative error is 100%
        return rep;
    }

    PIController pi;
    pi.kp         = kp;
    pi.ki         = ki;
    pi.integ_gate = cfg.integ_gate;
    pi.reset();

    double g = (initial_gain > 0.0) ? std::min(initial_gain, max_gain) : 1.0;

    /// Amplifier: y = g*x (Measured mode must rebuild the signal every iteration)
    const auto amplify = [&input](double gain) {
        ComplexVec y(input.size());
        for (std::size_t n = 0; n < input.size(); ++n)
            y[n] = input[n] * gain;
        return y;
    };

    double p_meas = 0.0;

    for (int k = 0; k < max_iter; ++k) {
        /// 1) Apply the amplifier + 2) measure the actual power
        if (cfg.sense == PowerSenseMode::Analytic) {
            p_meas = g * g * p_in;                 ///< O(1), no signal rebuild
        } else {
            p_meas = measureSignalPower(amplify(g)); ///< O(N), real measurement
        }
        if (history) history->push_back(p_meas);

        /// 3) Target power minus actual power
        const double err = target_power - p_meas;
        const double rel = std::abs(err) / target_power;

        rep.iterations = k + 1;
        rep.gain       = g;
        rep.power      = p_meas;
        rep.abs_error  = err;
        rep.rel_error  = rel;

        /// 4) Convergence check: relative error below the threshold
        if (rel < cfg.tol_rel) {
            rep.converged = true;
            break;
        }

        /// 5) Feed the error into the PI controller to get the power correction ratio u
        const double p_ref = std::max(p_meas, target_power * 1.0e-12); ///< Guard against division by zero
        const double e_rel = err / p_ref;              ///< Normalized power error
        const double u     = pi.update(e_rel);

        /// 6) Power correction -> gain correction (power is proportional to g^2, hence sqrt)
        const double scale = std::sqrt(std::max(0.0, 1.0 + u));
        g = std::min(g * scale, max_gain);
    }

    /// The output signal is rebuilt only once at the end (equivalent to per-iteration rebuild for a scalar gain)
    rep.gain   = g;
    rep.output = amplify(g);
    rep.power  = p_in * g * g;

    return rep;
}
