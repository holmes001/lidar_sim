/**
 * @file    app.cpp
 * @brief   通用小工具实现：FFT、取样、坐标轴、存图
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
 * @brief 单序列 FFT/IFFT 的公共实现
 * @param x    输入序列
 * @param sign FFTW_FORWARD 或 FFTW_BACKWARD
 * @return 变换结果，长度与输入相同，不做归一化
 * @note 每次调用都新建计划；批量场景请用 doppler.cpp 里 rangeDopplerMap 的一次性规划。
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
 * @brief 图片输出目录
 * @return <工程根>/figs
 * @note 以本文件路径为基准回退两级，因此无论从 build/ 还是工程根启动，
 *       图片都落到工程根的 figs/ 下。
 */
static std::filesystem::path fig_dir()
{
    const std::filesystem::path self(__FILE__);        ///< .../Src/app.cpp
    return self.parent_path().parent_path() / "figs";
}

/// 取复信号的实部
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
 * @brief 取复信号的模
 * @param u 复信号
 * @return 模信号
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
 * @brief 时间轴：t[i] = i·dt
 * @param n  点数
 * @param dt 采样间隔 [s]
 * @return 时间序列 [µs]
 * @note 图函数默认横轴是采样序号，横轴有物理含义时须用 PlotLineXY 显式传入。
 */
std::vector<double> time_axis(std::size_t n, double dt)
{
    std::vector<double> t(n);
    for (std::size_t i = 0; i < n; ++i)
        t[i] = static_cast<double>(i) * dt * 1e6;
    return t;
}

/**
 * @brief 按时间窗截取曲线
 * @param t_us  完整时间轴 [µs]
 * @param y     完整数据
 * @param t0    窗口起点 [µs]
 * @param t1    窗口终点 [µs]
 * @param t_out 截取后的时间轴
 * @param y_out 截取后的数据
 * @note 整段记录里信号只占很小一段（1 µs / 24 µs），不裁剪时波形会被压成一条竖线。
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
 * @brief 写盘
 * @param fig       已渲染的图像
 * @param file_path 绝对路径直接用；相对路径视为相对 <工程根>/figs，
 *                  目录不存在会自动创建
 * @return 保存成功返回 true
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
 * @brief 绘图并保存
 * @param a         待绘制的数据矩阵
 * @param st        绘图样式
 * @param file_path 保存路径，规则同 save_figure
 * @return 保存成功返回 true
 */
bool save_picture(const cv::Mat &a, const pl::Style &st, const std::string &file_path)
{
    return save_figure(pl::PlotLine(a, st), file_path);
}
