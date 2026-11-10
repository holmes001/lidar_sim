/**
 * @file    app.cpp
 * @brief   Generic helpers: FFT, sampling, axes, figure saving
 */
#include "app.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <fftw3.h>

/**
 * @brief Shared FFT/IFFT implementation for a single series
 * @param x    Input series
 * @param sign FFTW_FORWARD or FFTW_BACKWARD
 * @return Transform of the same length as the input, not normalized
 * @note A new plan is created on every call; for batch use, rely on the single plan in
 *       rangeDopplerMap (doppler.cpp).
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

/**
 * @brief Figure output directory
 * @return <project root>/figs
 * @note Two levels up from this file, so figures always land in the project root's
 *       figs/ whether the program starts from build/ or from the project root.
 */
static std::filesystem::path fig_dir()
{
    const std::filesystem::path self(__FILE__);        ///< .../Src/app.cpp
    return self.parent_path().parent_path() / "figs";
}

/// Real part of a complex signal
std::vector<double> vector_real(const ComplexVec &u)
{
    std::vector<double> v(u.size());
    for (std::size_t i = 0; i < u.size(); ++i)
    {
        v[i] = std::real(u[i]);
    }
    return v;
}


/**
 * @brief Magnitude of a complex signal
 * @param u Complex signal
 * @return Magnitude signal
 */
std::vector<double> vector_abs(const ComplexVec &u)
{
    std::vector<double> v(u.size());
    for (std::size_t i = 0; i < u.size(); ++i)
    {
        v[i] = std::abs(u[i]);
    }
    return v;
}

/**
 * @brief Time axis: t[i] = i*dt
 * @param n  Number of points
 * @param dt Sampling interval [s]
 * @return Time sequence [us]
 * @note Plot functions default to the sample index as x; use PlotLineXY whenever the
 *       axis has physical meaning.
 */
std::vector<double> time_axis(std::size_t n, double dt)
{
    std::vector<double> t(n);
    for (std::size_t i = 0; i < n; ++i)
        t[i] = static_cast<double>(i) * dt * 1e6;
    return t;
}

/**
 * @brief Crop a curve to a time window
 * @param t_us  Full time axis [us]
 * @param y     Full data
 * @param t0    Window start [us]
 * @param t1    Window end [us]
 * @param t_out Cropped time axis
 * @param y_out Cropped data
 * @note The signal occupies a small fraction of the record (1 us out of 24 us); without
 *       cropping the waveform is squeezed into a vertical line.
 */
void crop_window(const std::vector<double> &t_us, const std::vector<double> &y,
                        double t0, double t1,
                        std::vector<double> &t_out, std::vector<double> &y_out)
{
    t_out.clear();
    y_out.clear();
    const std::size_t n = std::min(t_us.size(), y.size());
    for (std::size_t i = 0; i < n; ++i)
    {
        if (t_us[i] >= t0 && t_us[i] <= t1)
        {
            t_out.push_back(t_us[i]);
            y_out.push_back(y[i]);
        }
    }
}

/**
 * @brief Write a figure to disk
 * @param fig       Rendered image
 * @param file_path An absolute path is used as is; a relative path is resolved against
 *                  <project root>/figs, and missing directories are created
 * @return true if the figure was saved
 */
bool save_figure(const cv::Mat &fig, const std::string &file_path)
{
    std::filesystem::path out(file_path);
    if (out.is_relative())
        out = fig_dir() / out;

    std::error_code ec;
    std::filesystem::create_directories(out.parent_path(), ec);

    if (pl::Save(fig, out.string()))
    {
        std::cout << "picture saved in " << out.string() << std::endl;
        return true;
    }
    std::cout << "save failed: " << out.string() << std::endl;
    return false;
}

/**
 * @brief Plot and save
 * @param a         Data matrix to draw
 * @param st        Plot style
 * @param file_path Output path, same rules as save_figure
 * @return true if the figure was saved
 */
bool save_picture(const cv::Mat &a, const pl::Style &st, const std::string &file_path)
{
    return save_figure(pl::PlotLine(a, st), file_path);
}
