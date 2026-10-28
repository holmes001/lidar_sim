/**
 * @file    filter.cpp
 * @brief   匹配滤波、FFT 与 range-Doppler 处理实现
 */
#include "filter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <fftw3.h>

#include "transport.h"

/**
 * @brief 匹配滤波：y = signal ⊛ h，h[k] = conj(ref[M-1-k])
 * @param signal                输入复信号
 * @param ref                   参考模板
 * @param mode                  输出长度与对齐方式
 * @param normalizeByRefEnergy  为 true 时按 ref 总能量归一化
 * @return 滤波后的复信号；任一输入为空时返回空
 */
ComplexVec matchedFilter(const ComplexVec& signal,
                         const ComplexVec& ref,
                         MatchFilterMode mode,
                         bool normalizeByRefEnergy)
{
    const std::size_t N = signal.size();
    const std::size_t M = ref.size();
    if (N == 0 || M == 0) return {};

    /// 直接时域卷积：y_full[n] = Σ_k signal[k]·conj(ref[M-1-(n-k)])
    ComplexVec full(N + M - 1, Complex(0.0, 0.0));
    for (std::size_t n = 0; n < N; ++n) {
        const Complex xn = signal[n];
        if (xn == Complex(0.0, 0.0)) continue;          ///< 稀疏信号优化
        for (std::size_t k = 0; k < M; ++k)
            full[n + k] += xn * std::conj(ref[M - 1 - k]);
    }

    /// 可选归一化：按参考信号总能量缩放，完全匹配时峰值 = 1
    if (normalizeByRefEnergy) {
        double e = 0.0;
        for (const auto& c : ref) e += std::norm(c);
        if (e > 0.0) {
            const double inv = 1.0 / e;
            for (auto& c : full) c *= inv;
        }
    }

    switch (mode) {
    case MatchFilterMode::Full:
        return full;

    case MatchFilterMode::Same: {
        /// 取 full[M-1 : M-1+N]，索引 t 对应信号从 t 起与 ref 匹配
        ComplexVec out(N);
        std::copy(full.begin() + (M - 1),
                  full.begin() + (M - 1 + N),
                  out.begin());
        return out;
    }

    case MatchFilterMode::Valid: {
        if (N < M) return {};                          ///< 无完全重叠区
        const std::size_t L = N - M + 1;
        ComplexVec out(L);
        std::copy(full.begin() + (M - 1),
                  full.begin() + (M - 1 + L),
                  out.begin());
        return out;
    }
    }
    return {};
}

/**
 * @brief 单序列 FFT/IFFT 的公共实现
 * @param x    输入序列
 * @param sign FFTW_FORWARD 或 FFTW_BACKWARD
 * @return 变换结果，长度与输入相同，不做归一化
 * @note 每次调用都新建计划；批量场景请用 rangeDopplerMap 里的一次性规划。
 */
static ComplexVec fftwTransform(const ComplexVec& x, int sign)
{
    ComplexVec out(x.size());
    if (x.empty()) return out;

    const std::size_t n = x.size();
    fftw_complex* in  = fftw_alloc_complex(n);
    fftw_complex* res = fftw_alloc_complex(n);
    if (!in || !res) { fftw_free(in); fftw_free(res); return out; }

    for (std::size_t i = 0; i < n; ++i) {
        in[i][0] = x[i].real();
        in[i][1] = x[i].imag();
    }

    fftw_plan plan = fftw_plan_dft_1d(static_cast<int>(n), in, res, sign, FFTW_ESTIMATE);
    fftw_execute(plan);

    for (std::size_t i = 0; i < n; ++i)
        out[i] = Complex(res[i][0], res[i][1]);

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(res);
    return out;
}

ComplexVec fft(const ComplexVec& x)
{
    return fftwTransform(x, FFTW_FORWARD);
}

ComplexVec ifft(const ComplexVec& x)
{
    ComplexVec out = fftwTransform(x, FFTW_BACKWARD);
    if (out.empty()) return out;

    const double inv = 1.0 / static_cast<double>(out.size());
    for (auto& c : out) c *= inv;
    return out;
}

ComplexVec fftShift(const ComplexVec& x)
{
    const std::size_t n = x.size();
    ComplexVec out(n);
    if (n == 0) return out;

    const std::size_t half = n / 2;
    for (std::size_t i = 0; i < n; ++i)
        out[i] = x[(i + half) % n];
    return out;
}

ComplexVec2D matchedFilterFrame(const ComplexVec2D& pulses,
                                const ComplexVec& ref,
                                MatchFilterMode mode)
{
    ComplexVec2D frame;
    if (pulses.empty() || ref.empty()) return frame;

    frame.reserve(pulses.size());
    for (const ComplexVec& p : pulses)
        frame.push_back(matchedFilter(p, ref, mode, false));

    return frame;
}

ComplexVec2D rangeDopplerMap(const ComplexVec2D& slow_time)
{
    if (slow_time.empty()) return ComplexVec2D();

    const std::size_t n_pulse = slow_time.size();
    const std::size_t n_range = slow_time[0].size();
    if (n_pulse == 0 || n_range == 0) return ComplexVec2D();

    /// 按列优先重排：同一距离门的 N 个慢时间样点连续存放，
    /// 这样一次批量规划就能算完所有距离门。
    fftw_complex* buf = fftw_alloc_complex(n_pulse * n_range);
    if (!buf) return ComplexVec2D();

    for (std::size_t r = 0; r < n_range; ++r) {
        for (std::size_t p = 0; p < n_pulse; ++p) {
            const Complex v = (p < slow_time.size() && r < slow_time[p].size())
                            ? slow_time[p][r] : Complex(0.0, 0.0);
            buf[r * n_pulse + p][0] = v.real();
            buf[r * n_pulse + p][1] = v.imag();
        }
    }

    const int n = static_cast<int>(n_pulse);
    const int howmany = static_cast<int>(n_range);
    fftw_plan plan = fftw_plan_many_dft(1, &n, howmany,
                                        buf, nullptr, 1, n,
                                        buf, nullptr, 1, n,
                                        FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    /// 顺带做 fftShift：行号 k 取原始 bin (k + N/2) % N，中心行即 0 Hz
    ComplexVec2D rd(n_pulse, ComplexVec(n_range));
    const std::size_t half = n_pulse / 2;
    for (std::size_t r = 0; r < n_range; ++r) {
        for (std::size_t k = 0; k < n_pulse; ++k) {
            const std::size_t src = (k + half) % n_pulse;
            rd[k][r] = Complex(buf[r * n_pulse + src][0], buf[r * n_pulse + src][1]);
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(buf);
    return rd;
}

std::vector<double> rangeAxis(std::size_t n_range, double dt)
{
    std::vector<double> axis(n_range, 0.0);
    for (std::size_t r = 0; r < n_range; ++r)
        axis[r] = static_cast<double>(r) * dt * c / 2.0;
    return axis;
}

std::vector<double> dopplerAxis(std::size_t n_pulse, double pri)
{
    std::vector<double> axis(n_pulse, 0.0);
    if (n_pulse == 0 || pri <= 0.0) return axis;

    const long n = static_cast<long>(n_pulse);
    const long half = n / 2;
    const double df = 1.0 / (static_cast<double>(n) * pri);
    for (long i = 0; i < n; ++i)
        axis[static_cast<std::size_t>(i)] = static_cast<double>(i - half) * df;
    return axis;
}

std::vector<double> velocityAxis(std::size_t n_pulse, double pri, double wavelength)
{
    std::vector<double> axis = dopplerAxis(n_pulse, pri);
    for (double& f : axis) f = f * wavelength / 2.0;
    return axis;
}

RangeDopplerPeak findRangeDopplerPeak(const ComplexVec2D& rd,
                                      double pri,
                                      double dt,
                                      double wavelength)
{
    RangeDopplerPeak peak;
    if (rd.empty() || rd[0].empty()) return peak;

    const std::size_t n_pulse = rd.size();
    const std::size_t n_range = rd[0].size();

    /// 1) 整数栅格上的最大值
    for (std::size_t k = 0; k < n_pulse; ++k) {
        for (std::size_t r = 0; r < n_range; ++r) {
            const double p = std::norm(rd[k][r]);
            if (p > peak.power) {
                peak.power = p;
                peak.doppler_index = k;
                peak.range_index   = r;
            }
        }
    }

    /// 2) 多普勒方向做三点抛物线插值，得到亚栅格峰值
    const std::size_t k0 = peak.doppler_index;
    double delta = 0.0;
    if (k0 > 0 && k0 + 1 < n_pulse) {
        const double ym = std::norm(rd[k0 - 1][peak.range_index]);
        const double y0 = std::norm(rd[k0][peak.range_index]);
        const double yp = std::norm(rd[k0 + 1][peak.range_index]);
        const double den = ym - 2.0 * y0 + yp;
        if (den != 0.0) delta = 0.5 * (ym - yp) / den;
        delta = std::max(-1.0, std::min(1.0, delta));
    }

    /// 3) 行号 -> 物理量
    const double df = (n_pulse > 1 && pri > 0.0)
                    ? 1.0 / (static_cast<double>(n_pulse) * pri) : 0.0;
    const double bin = static_cast<double>(k0) + delta
                     - static_cast<double>(n_pulse / 2);

    peak.doppler_hz = bin * df;
    peak.velocity   = peak.doppler_hz * wavelength / 2.0;
    peak.range_m    = static_cast<double>(peak.range_index) * dt * c / 2.0;
    return peak;
}
