/**
 * @file    plot.cpp
 * @brief   Implementation of plot.h
 *
 * @details
 * This file implements everything declared in `plot.h`. Only the functions declared
 * there are exported; the drawing primitives (painter, ticks, legend, ...) live in an
 * **anonymous namespace**, so they have internal linkage, stay out of the symbol table
 * and cannot clash with other translation units.
 *
 * @note OpenCV modules used: `core` (data structures), `imgproc` (applyColorMap /
 *       resize / drawing primitives), `imgcodecs` (only `Save()`, via imwrite).
 *
 * @note Two key implementation points:
 *       1. **Supersampling**: all drawing happens on a canvas `ss` times larger and is
 *          then downsampled with `INTER_AREA`, because `cv::line` has no anti-aliasing.
 *       2. **Translucent blending**: clone an overlay, fill the polygon on it, then
 *          `addWeighted(overlay, a, canvas, 1-a)`; outside the polygon the overlay is
 *          identical to the canvas, so those pixels are unchanged and this is equivalent
 *          to applying alpha to the polygon only.
 */

#include "plot.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace lidar
{
namespace plot
{

/* ========================================================================== */
/*                          Internal helpers (anonymous namespace)             */
/* ========================================================================== */

namespace
{

/** @brief Default palette (approximation of matplotlib tab10, **BGR order**) */
const std::vector<cv::Scalar> &Palette()
{
    static const std::vector<cv::Scalar> kPalette = {
        cv::Scalar(180, 119, 31),    // #1f77b4 blue
        cv::Scalar(14, 127, 255),    // #ff7f0e orange
        cv::Scalar(44, 160, 44),     // #2ca02c green
        cv::Scalar(40, 39, 214),     // #d62728 red
        cv::Scalar(189, 103, 148),   // #9467bd purple
        cv::Scalar(75, 86, 140),     // #8c564b brown
        cv::Scalar(194, 119, 227),   // #e377c2 pink
        cv::Scalar(127, 127, 127),   // #7f7f7f gray
        cv::Scalar(188, 189, 34),    // #bcbd22 olive
        cv::Scalar(23, 190, 207)     // #17becf cyan
    };
    return kPalette;
}

/** @brief Get the i-th palette color (wrapping around) */
cv::Scalar ColorAt(int i)
{
    const auto &p = Palette();
    return p[static_cast<size_t>(i) % p.size()];
}

/** @brief Whether a value is a valid point (not NaN / not Inf) */
bool Finite(double v)
{
    return std::isfinite(v);
}

/**
 * @brief Convert any depth/channel `cv::Mat` into a single-channel `CV_64F`
 * @param m Input (CV_8U/16U/32S/32F/64F supported; multi-channel takes channel 0)
 * @return New CV_64F matrix (deep copy)
 */
cv::Mat ToF64(const cv::Mat &m)
{
    if (m.empty())
    {
        return cv::Mat();
    }

    cv::Mat src = m;
    if (src.channels() > 1)
    {
        std::vector<cv::Mat> ch;
        cv::split(src, ch);
        src = ch[0];
    }
    if (src.depth() != CV_64F)
    {
        src.convertTo(src, CV_64F);
    }
    else
    {
        src = src.clone();
    }
    return src;
}

/** @brief Whether the matrix is "1-D" (1xN or Nx1) */
bool IsVector(const cv::Mat &m)
{
    return (m.rows == 1) || (m.cols == 1);
}

/** @brief Flatten a 1-D matrix into a `std::vector<double>` (row-major) */
std::vector<double> Flatten(const cv::Mat &m)
{
    std::vector<double> out;
    out.reserve(static_cast<size_t>(m.total()));
    for (int r = 0; r < m.rows; ++r)
    {
        for (int c = 0; c < m.cols; ++c)
        {
            out.push_back(m.at<double>(r, c));
        }
    }
    return out;
}

/**
 * @brief Split a 2-D matrix into "one series per row"
 * @param m Input (a 1-D input yields a single series)
 * @return List of series, each as long as the number of samples
 */
std::vector<std::vector<double>> SplitSeries(const cv::Mat &m)
{
    std::vector<std::vector<double>> series;

    if (m.empty())
    {
        return series;
    }

    if (IsVector(m))
    {
        series.push_back(Flatten(m));
        return series;
    }

    series.resize(static_cast<size_t>(m.rows));
    for (int r = 0; r < m.rows; ++r)
    {
        series[static_cast<size_t>(r)].resize(static_cast<size_t>(m.cols));
        for (int c = 0; c < m.cols; ++c)
        {
            series[static_cast<size_t>(r)][static_cast<size_t>(c)] = m.at<double>(r, c);
        }
    }
    return series;
}

/**
 * @brief "Nice" number step (Heckbert's nice-number algorithm)
 * @param range Span to cover (positive)
 * @param round Whether to round to 1/2/5/10
 * @return Step size
 * @note Keeps ticks on 1/2/5 * 10^n instead of values like 0.37, which is the key
 *       detail behind a clean publication figure.
 */
double NiceNum(double range, bool round)
{
    if (range <= 0.0 || !std::isfinite(range))
    {
        return 1.0;
    }

    const double expv = std::floor(std::log10(range));
    const double f    = range / std::pow(10.0, expv);

    double nf;
    if (round)
    {
        nf = (f < 1.5) ? 1.0 : ((f < 3.0) ? 2.0 : ((f < 7.0) ? 5.0 : 10.0));
    }
    else
    {
        nf = (f <= 1.0) ? 1.0 : ((f <= 2.0) ? 2.0 : ((f <= 5.0) ? 5.0 : 10.0));
    }

    return nf * std::pow(10.0, expv);
}

/**
 * @brief Generate nice ticks covering [lo, hi]
 * @param lo   Lower bound
 * @param hi   Upper bound
 * @param want Desired tick count (hint)
 * @param step Returns the actual step
 * @return List of tick values
 */
std::vector<double> Ticks(double lo, double hi, int want, double &step)
{
    std::vector<double> out;

    if (!std::isfinite(lo) || !std::isfinite(hi) || hi <= lo)
    {
        step = 1.0;
        out.push_back(lo);
        return out;
    }

    if (want < 2)
    {
        want = 2;
    }

    const double raw   = (hi - lo) / static_cast<double>(want);
    step               = NiceNum(raw, true);
    const double first = std::ceil(lo / step) * step;

    for (double v = first; v <= hi + step * 1e-6; v += step)
    {
        out.push_back((std::fabs(v) < step * 1e-9) ? 0.0 : v);
        if (out.size() > 256U)
        {
            break;    /* Defensive: avoid an infinite loop if the step degenerates */
        }
    }
    return out;
}

/** @brief Number to string; `decimals < 0` uses the short "%.4g" format */
std::string NumToStr(double v, int decimals)
{
    char buf[64];

    if (decimals >= 0)
    {
        std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    }
    else
    {
        std::snprintf(buf, sizeof(buf), "%.4g", v);
    }
    return std::string(buf);
}

/**
 * @brief Linear mapping from data domain to pixel domain
 * @note Pass `p0 = bottom` and `p1 = top` for y to flip the pixel axis.
 */
struct LinearAxis
{
    double lo = 0.0;    /**< Data lower bound */
    double hi = 1.0;    /**< Data upper bound */
    double p0 = 0.0;    /**< Pixel coordinate of lo */
    double p1 = 1.0;    /**< Pixel coordinate of hi */

    /** @brief Map a data value to a pixel coordinate */
    double Map(double v) const
    {
        const double d = (hi - lo);
        if (std::fabs(d) < 1e-12)
        {
            return 0.5 * (p0 + p1);
        }
        return p0 + (v - lo) / d * (p1 - p0);
    }
};

/**
 * @brief Painter on the supersampled canvas: all coordinates are **logical**, scaled by ss
 *        internally
 *
 * @details Collecting every primitive in one class lets the upper layers work purely in
 * logical coordinates while scaling by the supersample factor and swapping the text
 * renderer happen in one place.
 */
class Painter
{
public:
    /**
     * @param canvas Supersampled target canvas
     * @param ss     Supersample factor
     * @param tr     Text renderer (may be nullptr)
     */
    Painter(cv::Mat &canvas, int ss, const TextRenderer &tr)
        : img_(canvas), ss_(ss), tr_(tr)
    {
    }

    /** @brief Logical coordinates -> pixel coordinates */
    cv::Point P(double x, double y) const
    {
        return cv::Point(cvRound(x * ss_), cvRound(y * ss_));
    }

    /** @brief Draw a line segment */
    void Line(double x0, double y0, double x1, double y1, cv::Scalar color, double w)
    {
        cv::line(img_, P(x0, y0), P(x1, y1), color,
                 std::max(1, cvRound(w * ss_)), cv::LINE_8);
    }

    /** @brief Draw a polyline (logical-coordinate point list) */
    void Polyline(const std::vector<cv::Point2d> &pts, cv::Scalar color, double w)
    {
        if (pts.size() < 2U)
        {
            return;
        }

        std::vector<cv::Point> ip;
        ip.reserve(pts.size());
        for (const auto &p : pts)
        {
            ip.push_back(cv::Point(cvRound(p.x * ss_), cvRound(p.y * ss_)));
        }
        cv::polylines(img_, ip, false, color, std::max(1, cvRound(w * ss_)), cv::LINE_8);
    }

    /** @brief Fill a polygon (logical-coordinate point list) */
    void FillPoly(const std::vector<cv::Point2d> &pts, cv::Scalar color)
    {
        if (pts.size() < 3U)
        {
            return;
        }

        std::vector<cv::Point> ip;
        ip.reserve(pts.size());
        for (const auto &p : pts)
        {
            ip.push_back(cv::Point(cvRound(p.x * ss_), cvRound(p.y * ss_)));
        }

        const std::vector<std::vector<cv::Point>> polys{ip};
        cv::fillPoly(img_, polys, color, cv::LINE_8);
    }

    /** @brief Draw a filled dot */
    void Dot(double x, double y, int radius, cv::Scalar color)
    {
        cv::circle(img_, P(x, y), std::max(1, radius * ss_), color, cv::FILLED, cv::LINE_8);
    }

    /** @brief Draw a rectangle outline */
    void Rect(double x, double y, double w, double h, cv::Scalar color, double lw)
    {
        cv::rectangle(img_, P(x, y), P(x + w, y + h), color,
                      std::max(1, cvRound(lw * ss_)), cv::LINE_8);
    }

    /** @brief Fill a rectangle */
    void FillRect(double x, double y, double w, double h, cv::Scalar color)
    {
        cv::rectangle(img_, P(x, y), P(x + w, y + h), color, cv::FILLED);
    }

    /**
     * @brief Draw text (logical coordinates + alignment)
     * @note Text size is estimated with `cv::getTextSize`. A custom renderer is aligned
     *       using the same estimate, which is good enough for monospaced or nearly
     *       monospaced fonts; proportional fonts may drift slightly.
     */
    void Text(const std::string &text, double x, double y, double scale,
              cv::Scalar color, Align ha = Align::Left, VAlign va = VAlign::Middle)
    {
        if (text.empty())
        {
            return;
        }

        const double scaled = scale * ss_;
        const int    th     = std::max(1, cvRound(1.0 * ss_));

        int base = 0;
        const cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scaled, th, &base);

        double ox = x;
        double oy = y;

        if (ha == Align::Center)
        {
            ox -= 0.5 * sz.width / ss_;
        }
        else if (ha == Align::Right)
        {
            ox -= static_cast<double>(sz.width) / ss_;
        }

        if (va == VAlign::Middle)
        {
            oy += 0.5 * (sz.height - base) / ss_;
        }
        else if (va == VAlign::Top)
        {
            oy += static_cast<double>(sz.height) / ss_;
        }

        const cv::Point org(cvRound(ox * ss_), cvRound(oy * ss_));

        if (tr_)
        {
            tr_(img_, text, org, scaled, color, th);
        }
        else
        {
            cv::putText(img_, text, org, cv::FONT_HERSHEY_SIMPLEX, scaled, color, th, cv::LINE_8);
        }
    }

    /** @brief Access the underlying canvas */
    cv::Mat &Canvas() { return img_; }

    /** @brief Supersample factor */
    int Scale() const { return ss_; }

private:
    cv::Mat            &img_;
    int                 ss_;
    const TextRenderer &tr_;
};

/**
 * @brief Statistics: min/max over valid points (NaN/Inf skipped)
 * @param series Set of series
 * @param lo     Output minimum
 * @param hi     Output maximum
 * @return Number of valid points
 */
size_t MinMax(const std::vector<std::vector<double>> &series, double &lo, double &hi)
{
    lo = std::numeric_limits<double>::max();
    hi = std::numeric_limits<double>::lowest();
    size_t n = 0U;

    for (const auto &s : series)
    {
        for (double v : s)
        {
            if (!Finite(v))
            {
                continue;
            }
            lo = std::min(lo, v);
            hi = std::max(hi, v);
            ++n;
        }
    }

    if (n == 0U)
    {
        lo = 0.0;
        hi = 1.0;
    }
    return n;
}

/**
 * @brief Draw grid, frame, ticks and tick labels inside a rectangle
 * @param rx,ry,rw,rh Plot area (logical coordinates)
 * @param xlo,xhi,y_lo,y_hi Data domains
 */
void DrawGridAndFrame(Painter &pt, const Style &st,
                      double rx, double ry, double rw, double rh,
                      double xlo, double xhi, double y_lo, double y_hi)
{
    double sx = 0.0;
    double sy = 0.0;

    const bool auto_xstep = (st.x_tick_step <= 0.0);
    const bool auto_ystep = (st.y_tick_step <= 0.0);

    const std::vector<double> xt = auto_xstep ? Ticks(xlo, xhi, 7, sx) : std::vector<double>();
    const std::vector<double> yt = auto_ystep ? Ticks(y_lo, y_hi, 6, sy) : std::vector<double>();

    LinearAxis ax;
    ax.lo = xlo; ax.hi = xhi; ax.p0 = rx; ax.p1 = rx + rw;

    LinearAxis ay;
    ay.lo = y_lo; ay.hi = y_hi; ay.p0 = ry + rh; ay.p1 = ry;   /* Pixel y axis points down */

    const double xstep = auto_xstep ? sx : st.x_tick_step;
    const double ystep = auto_ystep ? sy : st.y_tick_step;

    /* ---- Grid ---- */
    if (st.grid)
    {
        if (auto_xstep)
        {
            for (double v : xt)
            {
                pt.Line(ax.Map(v), ry, ax.Map(v), ry + rh, st.grid_color, 1.0);
            }
        }
        else
        {
            for (double v = std::ceil(xlo / xstep) * xstep; v <= xhi; v += xstep)
            {
                pt.Line(ax.Map(v), ry, ax.Map(v), ry + rh, st.grid_color, 1.0);
            }
        }

        if (auto_ystep)
        {
            for (double v : yt)
            {
                pt.Line(rx, ay.Map(v), rx + rw, ay.Map(v), st.grid_color, 1.0);
            }
        }
        else
        {
            for (double v = std::ceil(y_lo / ystep) * ystep; v <= y_hi; v += ystep)
            {
                pt.Line(rx, ay.Map(v), rx + rw, ay.Map(v), st.grid_color, 1.0);
            }
        }
    }

    /* ---- Frame ---- */
    pt.Rect(rx, ry, rw, rh, st.frame_color, 1.2);

    /* ---- Ticks + tick labels ---- */
    const double tsc = st.font_scale * 0.82;

    auto one_x = [&](double v) {
        const double px = ax.Map(v);
        pt.Line(px, ry + rh, px, ry + rh + st.tick_len, st.foreground, 1.2);
        pt.Text(NumToStr(v, st.decimals), px, ry + rh + st.tick_len + 4.0,
                tsc, st.foreground, Align::Center, VAlign::Top);
    };

    auto one_y = [&](double v) {
        const double py = ay.Map(v);
        pt.Line(rx - st.tick_len, py, rx, py, st.foreground, 1.2);
        pt.Text(NumToStr(v, st.decimals), rx - st.tick_len - 4.0, py,
                tsc, st.foreground, Align::Right, VAlign::Middle);
    };

    if (auto_xstep)
    {
        for (double v : xt) { one_x(v); }
    }
    else
    {
        for (double v = std::ceil(xlo / xstep) * xstep; v <= xhi; v += xstep) { one_x(v); }
    }

    if (auto_ystep)
    {
        for (double v : yt) { one_y(v); }
    }
    else
    {
        for (double v = std::ceil(y_lo / ystep) * ystep; v <= y_hi; v += ystep) { one_y(v); }
    }
}

/** @brief Draw the title (centered horizontally on the whole figure) */
void DrawTitle(Painter &pt, const Style &st, double cx)
{
    if (st.title.empty())
    {
        return;
    }
    pt.Text(st.title, cx, st.margin_top * 0.45, st.font_scale * 1.18,
            st.foreground, Align::Center, VAlign::Middle);
}

/**
 * @brief Draw the axis labels (the y label is rotated 90 degrees)
 * @note OpenCV has no text rotation API, so the text is drawn to a small temporary image,
 *       rotated with `cv::rotate`, and then only its non-background pixels are copied
 *       back onto the main canvas.
 */
void DrawAxisLabels(Painter &pt, const Style &st,
                    double rx, double ry, double rw, double rh)
{
    if (!st.xlabel.empty())
    {
        pt.Text(st.xlabel, rx + 0.5 * rw, st.height - 16.0,
                st.font_scale * 0.95, st.foreground, Align::Center, VAlign::Bottom);
    }

    if (st.ylabel.empty())
    {
        return;
    }

    const double scaled = st.font_scale * 0.95 * pt.Scale();
    const int    th     = std::max(1, cvRound(1.0 * pt.Scale()));

    int base = 0;
    const cv::Size sz = cv::getTextSize(st.ylabel, cv::FONT_HERSHEY_SIMPLEX, scaled, th, &base);

    const int pad = cvRound(4.0 * pt.Scale());
    cv::Mat tmp(sz.height + base + 6 * pad, sz.width + 4 * pad, CV_8UC3, st.background);

    if (st.text_renderer)
    {
        st.text_renderer(tmp, st.ylabel, cv::Point(2 * pad, 2 * pad + sz.height), scaled,
                         st.foreground, th);
    }
    else
    {
        cv::putText(tmp, st.ylabel, cv::Point(2 * pad, 2 * pad + sz.height),
                    cv::FONT_HERSHEY_SIMPLEX, scaled, st.foreground, th, cv::LINE_8);
    }

    cv::Mat rot;
    cv::rotate(tmp, rot, cv::ROTATE_90_COUNTERCLOCKWISE);

    const cv::Vec3b bg(static_cast<uchar>(st.background[0]),
                       static_cast<uchar>(st.background[1]),
                       static_cast<uchar>(st.background[2]));

    const double cx = 14.0 * pt.Scale();
    const double cy = (ry + 0.5 * rh) * pt.Scale();

    for (int r = 0; r < rot.rows; ++r)
    {
        const int ty = cvRound(cy - rot.rows * 0.5) + r;
        if (ty < 0 || ty >= pt.Canvas().rows)
        {
            continue;
        }
        for (int c = 0; c < rot.cols; ++c)
        {
            const int tx = cvRound(cx - rot.cols * 0.5) + c;
            if (tx < 0 || tx >= pt.Canvas().cols)
            {
                continue;
            }
            if (rot.at<cv::Vec3b>(r, c) == bg)
            {
                continue;    /* Skip background pixels */
            }
            pt.Canvas().at<cv::Vec3b>(ty, tx) = rot.at<cv::Vec3b>(r, c);
        }
    }
}

/** @brief Draw the legend (white box + thin border + color swatch + text) */
void DrawLegend(Painter &pt, const Style &st,
                const std::vector<std::string> &names,
                const std::vector<cv::Scalar> &colors,
                double rx, double ry, double rw, double rh)
{
    if (names.empty() || names.size() != colors.size())
    {
        return;
    }

    const double fs  = st.font_scale * 0.86;
    const double lh  = std::max(18.0, 26.0 * fs);      /* Row height */
    const double sw  = 26.0;                          /* Swatch width */
    const double gap = 8.0;

    int base = 0;
    double maxw = 0.0;
    for (const auto &s : names)
    {
        if (s.empty())
        {
            continue;
        }
        const cv::Size sz = cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX,
                                            fs * pt.Scale(), 1, &base);
        maxw = std::max(maxw, static_cast<double>(sz.width) / pt.Scale());
    }
    if (maxw <= 0.0)
    {
        return;
    }

    const double boxw = sw + gap + maxw + 16.0;
    const double boxh = lh * static_cast<double>(names.size()) + 10.0;

    double bx;
    double by;

    if (st.legend_top_right)
    {
        bx = rx + rw - boxw - 10.0;
        by = ry + 10.0;
    }
    else
    {
        bx = rx + 0.5 * (rw - boxw);
        by = ry + rh + 10.0;
    }

    pt.FillRect(bx, by, boxw, boxh, st.background);
    pt.Rect(bx, by, boxw, boxh, st.grid_color, 1.0);

    for (size_t i = 0; i < names.size(); ++i)
    {
        if (names[i].empty())
        {
            continue;
        }

        const double ly = by + 5.0 + lh * static_cast<double>(i) + lh * 0.5;

        pt.Line(bx + 8.0, ly, bx + 8.0 + sw, ly, colors[i], st.line_width);
        pt.Text(names[i], bx + 8.0 + sw + gap, ly, fs, st.foreground,
                Align::Left, VAlign::Middle);
    }
}

/** @brief Create a canvas at the supersample factor and fill it with the background color */
cv::Mat NewCanvas(const Style &st)
{
    const int w = std::max(160, st.width) * std::max(1, st.supersample);
    const int h = std::max(120, st.height) * std::max(1, st.supersample);
    cv::Mat img(h, w, CV_8UC3, st.background);
    return img;
}

/** @brief Downsample the supersampled canvas back to logical size */
cv::Mat Finish(const cv::Mat &canvas, const Style &st)
{
    cv::Mat out;
    const int ss = std::max(1, st.supersample);
    if (ss == 1)
    {
        out = canvas.clone();
    }
    else
    {
        cv::resize(canvas, out,
                   cv::Size(std::max(160, st.width), std::max(120, st.height)),
                   0.0, 0.0, cv::INTER_AREA);
    }
    return out;
}

}   // namespace

/* ========================================================================== */
/*                          Input data helpers                                */
/* ========================================================================== */

cv::Mat Make1D(const std::vector<double> &v)
{
    cv::Mat m(static_cast<int>(v.size()), 1, CV_64F);
    for (int i = 0; i < m.rows; ++i)
    {
        m.at<double>(i, 0) = v[static_cast<size_t>(i)];
    }
    return m;
}

cv::Mat Make2D(const std::vector<std::vector<double>> &v)
{
    if (v.empty())
    {
        return cv::Mat();
    }

    const size_t cols = v.front().size();
    for (const auto &row : v)
    {
        if (row.size() != cols)
        {
            throw std::invalid_argument("Make2D: rows have different lengths");
        }
    }

    cv::Mat m(static_cast<int>(v.size()), static_cast<int>(cols), CV_64F);
    for (int r = 0; r < m.rows; ++r)
    {
        for (int c = 0; c < m.cols; ++c)
        {
            m.at<double>(r, c) = v[static_cast<size_t>(r)][static_cast<size_t>(c)];
        }
    }
    return m;
}

/* ========================================================================== */
/*                                 Line plot                                  */
/* ========================================================================== */

/* ========================================================================== */
/*          Line plot core (internal, shared by the two public entries)       */
/* ========================================================================== */

namespace
{

/**
 * @brief Line plot core implementation
 * @param px x array; **nullptr means use the sample index 0..N-1**
 * @param series y values (1-D = single curve, 2-D = multiple curves)
 * @param st Style
 * @param series_in_cols Whether a 2-D input is split by column
 * @return BGR image
 * @note If x is shorter than the sample count it falls back to the index axis (the entry
 *       points validate this beforehand).
 */
cv::Mat PlotLineImpl(const std::vector<double> *px,
                     const cv::Mat &series,
                     const Style &st,
                     bool series_in_cols)
{
    cv::Mat m = ToF64(series);
    if (m.empty())
    {
        return Finish(NewCanvas(st), st);
    }

    std::vector<std::vector<double>> all;

    if (IsVector(m))
    {
        all.push_back(Flatten(m));
    }
    else if (series_in_cols)
    {
        cv::Mat t;
        cv::transpose(m, t);
        all = SplitSeries(t);
    }
    else
    {
        all = SplitSeries(m);
    }

    size_t npts = 0U;
    for (const auto &s : all)
    {
        npts = std::max(npts, s.size());
    }
    if (npts == 0U)
    {
        return Finish(NewCanvas(st), st);
    }

    /* Fall back to the index axis when x is too short, avoiding out-of-bounds access */
    if ((px != nullptr) && (px->size() < npts))
    {
        px = nullptr;
    }

    /* ---- y range ---- */
    double ylo = 0.0;
    double yhi = 0.0;
    MinMax(all, ylo, yhi);

    if (st.y_include_zero)
    {
        ylo = std::min(ylo, 0.0);
        yhi = std::max(yhi, 0.0);
    }
    if (!st.auto_y && st.y_max > st.y_min)
    {
        ylo = st.y_min;
        yhi = st.y_max;
    }
    if (yhi - ylo < 1e-12)
    {
        ylo -= 0.5;
        yhi += 0.5;
    }
    const double pad = (yhi - ylo) * st.y_pad_ratio;
    ylo -= pad;
    yhi += pad;

    /* ---- x range ---- */
    double xlo = 0.0;
    double xhi = 1.0;

    if (px != nullptr)
    {
        xlo = std::numeric_limits<double>::max();
        xhi = std::numeric_limits<double>::lowest();

        for (size_t k = 0; k < npts; ++k)
        {
            const double xv = (*px)[k];
            if (!Finite(xv))
            {
                continue;
            }
            xlo = std::min(xlo, xv);
            xhi = std::max(xhi, xv);
        }
        if (xlo > xhi)
        {
            xlo = 0.0;
            xhi = 1.0;
        }
        if (!st.auto_x && st.x_max > st.x_min)
        {
            xlo = st.x_min;
            xhi = st.x_max;
        }
        if (xhi - xlo < 1e-12)
        {
            xlo -= 0.5;
            xhi += 0.5;
        }
    }
    else
    {
        xlo = st.auto_x ? 0.0 : st.x_min;
        xhi = st.auto_x ? static_cast<double>(npts - 1U) : st.x_max;
    }

    /* ---- Canvas ---- */
    cv::Mat canvas = NewCanvas(st);
    Painter pt(canvas, std::max(1, st.supersample), st.text_renderer);

    const double rx = st.margin_left;
    const double ry = st.margin_top;
    const double rw = st.width - st.margin_left - st.margin_right;
    const double rh = st.height - st.margin_top - st.margin_bottom;

    DrawGridAndFrame(pt, st, rx, ry, rw, rh, xlo, xhi, ylo, yhi);

    LinearAxis ax;
    ax.lo = xlo; ax.hi = xhi; ax.p0 = rx; ax.p1 = rx + rw;

    LinearAxis ay;
    ay.lo = ylo; ay.hi = yhi; ay.p0 = ry + rh; ay.p1 = ry;

    /* ---- Curves ---- */
    for (size_t i = 0; i < all.size(); ++i)
    {
        const cv::Scalar color = ColorAt(static_cast<int>(i));
        const size_t     n     = all[i].size();
        std::vector<cv::Point2d> run;

        auto flush = [&]() {
            if (run.size() >= 2U)
            {
                pt.Polyline(run, color, st.line_width);
            }
            run.clear();
        };

        for (size_t k = 0; k < n; ++k)
        {
            const double v = all[i][k];
            if (!Finite(v))
            {
                flush();                       /* NaN breaks the curve */
                continue;
            }

            double xv;
            if (px != nullptr)
            {
                xv = (*px)[k];
                if (!Finite(xv))
                {
                    flush();                   /* Missing x also breaks the curve */
                    continue;
                }
            }
            else
            {
                /* Index axis: map k linearly onto [xlo, xhi] */
                xv = ((xhi - xlo) > 1e-12)
                         ? (xlo + (xhi - xlo) * static_cast<double>(k) /
                                      static_cast<double>(std::max<size_t>(1U, n - 1U)))
                         : xlo;
            }

            const double pixel_x = ax.Map(xv);
            run.push_back(cv::Point2d(pixel_x, ay.Map(v)));

            if (st.show_markers)
            {
                pt.Dot(pixel_x, ay.Map(v), st.marker_radius, color);
            }
        }
        flush();
    }

    /* ---- Annotations ---- */
    DrawTitle(pt, st, 0.5 * st.width);

    {
        std::vector<std::string> names;
        std::vector<cv::Scalar>  colors;
        for (size_t i = 0; i < all.size(); ++i)
        {
            names.push_back(i < st.legend.size() ? st.legend[i] : std::string());
            colors.push_back(ColorAt(static_cast<int>(i)));
        }
        DrawLegend(pt, st, names, colors, rx, ry, rw, rh);
    }

    DrawAxisLabels(pt, st, rx, ry, rw, rh);

    return Finish(canvas, st);
}

}   // namespace

cv::Mat PlotLine(const cv::Mat &series, const Style &st, bool series_in_cols)
{
    return PlotLineImpl(nullptr, series, st, series_in_cols);
}

cv::Mat PlotLineXY(const cv::Mat &x, const cv::Mat &series,
                   const Style &st, bool series_in_cols)
{
    /* ---- Validate the x axis ---- */
    cv::Mat xm = ToF64(x);
    if (xm.empty())
    {
        /* No valid x given, so fall back to the index axis and still produce a figure */
        return PlotLineImpl(nullptr, series, st, series_in_cols);
    }

    const std::vector<double> xv = Flatten(xm);

    /* ---- Validate the length: x must cover every sample ---- */
    cv::Mat sm = ToF64(series);
    if (sm.empty())
    {
        return Finish(NewCanvas(st), st);
    }

    const size_t npts = IsVector(sm) ? static_cast<size_t>(sm.total())
                                     : static_cast<size_t>(sm.cols);

    if (xv.size() < npts)
    {
        return Finish(NewCanvas(st), st);      /* Length mismatch: return a blank image */
    }

    return PlotLineImpl(&xv, series, st, series_in_cols);
}

/* ========================================================================== */
/*                        Mean +/- std band line plot                         */
/* ========================================================================== */

cv::Mat PlotLineBand(const cv::Mat &samples, const Style &st, double k_sigma)
{
    cv::Mat m = ToF64(samples);
    if (m.empty() || m.rows < 2)
    {
        return Finish(NewCanvas(st), st);
    }

    const int cols = m.cols;

    std::vector<double> mean(static_cast<size_t>(cols), std::numeric_limits<double>::quiet_NaN());
    std::vector<double> lo(static_cast<size_t>(cols), std::numeric_limits<double>::quiet_NaN());
    std::vector<double> hi(static_cast<size_t>(cols), std::numeric_limits<double>::quiet_NaN());

    double gmin = std::numeric_limits<double>::max();
    double gmax = std::numeric_limits<double>::lowest();

    for (int c = 0; c < cols; ++c)
    {
        double sum  = 0.0;
        double sum2 = 0.0;
        int    n    = 0;

        for (int r = 0; r < m.rows; ++r)
        {
            const double v = m.at<double>(r, c);
            if (!Finite(v))
            {
                continue;
            }
            sum  += v;
            sum2 += v * v;
            ++n;
        }

        if (n == 0)
        {
            continue;
        }

        const double mu  = sum / static_cast<double>(n);
        double       var = sum2 / static_cast<double>(n) - mu * mu;
        if (var < 0.0)
        {
            var = 0.0;                     /* Guard against numerical error */
        }
        const double sd = std::sqrt(var);

        mean[static_cast<size_t>(c)] = mu;
        lo[static_cast<size_t>(c)]   = mu - k_sigma * sd;
        hi[static_cast<size_t>(c)]   = mu + k_sigma * sd;

        gmin = std::min(gmin, lo[static_cast<size_t>(c)]);
        gmax = std::max(gmax, hi[static_cast<size_t>(c)]);
    }

    if (gmin > gmax)
    {
        gmin = 0.0;
        gmax = 1.0;
    }

    if (st.y_include_zero)
    {
        gmin = std::min(gmin, 0.0);
        gmax = std::max(gmax, 0.0);
    }
    if (!st.auto_y && st.y_max > st.y_min)
    {
        gmin = st.y_min;
        gmax = st.y_max;
    }
    if (gmax - gmin < 1e-12)
    {
        gmin -= 0.5;
        gmax += 0.5;
    }
    const double pad = (gmax - gmin) * st.y_pad_ratio;
    gmin -= pad;
    gmax += pad;

    const double xlo = 0.0;
    const double xhi = static_cast<double>(std::max(1, cols - 1));

    cv::Mat canvas = NewCanvas(st);
    Painter pt(canvas, std::max(1, st.supersample), st.text_renderer);

    const double rx = st.margin_left;
    const double ry = st.margin_top;
    const double rw = st.width - st.margin_left - st.margin_right;
    const double rh = st.height - st.margin_top - st.margin_bottom;

    LinearAxis ax;
    ax.lo = xlo; ax.hi = xhi; ax.p0 = rx; ax.p1 = rx + rw;

    LinearAxis ay;
    ay.lo = gmin; ay.hi = gmax; ay.p0 = ry + rh; ay.p1 = ry;

    const cv::Scalar c0 = ColorAt(0);

    /* ---- 1. Error band (draw the band before the grid and the line) ---- */
    if (st.band_alpha > 0.0)
    {
        std::vector<cv::Point2d> poly;
        poly.reserve(static_cast<size_t>(cols) * 2U);

        auto x_at = [&](int c) {
            return ax.Map(xlo + (xhi - xlo) * static_cast<double>(c) / std::max(1.0, xhi));
        };

        for (int c = 0; c < cols; ++c)
        {
            const double v = hi[static_cast<size_t>(c)];
            if (!Finite(v))
            {
                continue;
            }
            poly.push_back(cv::Point2d(x_at(c), ay.Map(v)));
        }
        for (int c = cols - 1; c >= 0; --c)
        {
            const double v = lo[static_cast<size_t>(c)];
            if (!Finite(v))
            {
                continue;
            }
            poly.push_back(cv::Point2d(x_at(c), ay.Map(v)));
        }

        if (poly.size() >= 3U)
        {
            cv::Mat overlay = canvas.clone();
            Painter op(overlay, std::max(1, st.supersample), st.text_renderer);
            op.FillPoly(poly, c0);
            cv::addWeighted(overlay, st.band_alpha, canvas, 1.0 - st.band_alpha, 0.0, canvas);
        }
    }

    /* ---- 2. Grid / frame / ticks ---- */
    DrawGridAndFrame(pt, st, rx, ry, rw, rh, xlo, xhi, gmin, gmax);

    /* ---- 3. Mean line ---- */
    {
        std::vector<cv::Point2d> run;
        auto flush = [&]() {
            if (run.size() >= 2U) { pt.Polyline(run, c0, st.line_width); }
            run.clear();
        };

        for (int c = 0; c < cols; ++c)
        {
            const double v = mean[static_cast<size_t>(c)];
            if (!Finite(v))
            {
                flush();
                continue;
            }
            run.push_back(cv::Point2d(
                ax.Map(xlo + (xhi - xlo) * static_cast<double>(c) / std::max(1.0, xhi)),
                ay.Map(v)));
        }
        flush();
    }

    /* ---- 4. Annotations ---- */
    DrawTitle(pt, st, 0.5 * st.width);

    {
        std::vector<std::string> names;
        std::vector<cv::Scalar>  cols4;

        /* The default legend name must be **pure ASCII**: cv::putText only supports ASCII,
         * so non-ASCII symbols render as "??" (see "Non-ASCII label limitation" in plot.h). */
        if (st.legend.empty())
        {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "mean +/- %.1f sigma", k_sigma);
            names.push_back(std::string(buf));
        }
        else
        {
            names.push_back(st.legend[0]);
        }

        cols4.push_back(c0);
        DrawLegend(pt, st, names, cols4, rx, ry, rw, rh);
    }

    DrawAxisLabels(pt, st, rx, ry, rw, rh);

    return Finish(canvas, st);
}

/* ========================================================================== */
/*                                Scatter plot                                */
/* ========================================================================== */

cv::Mat PlotScatter(const cv::Mat &xy, const Style &st, bool points_in_rows)
{
    cv::Mat m = ToF64(xy);
    if (m.empty())
    {
        return Finish(NewCanvas(st), st);
    }

    /* Normalize to 2xN */
    cv::Mat t;
    if (points_in_rows)
    {
        cv::transpose(m, t);       /* N×2 -> 2×N */
    }
    else
    {
        t = m;
    }

    if (t.rows < 2)
    {
        return Finish(NewCanvas(st), st);
    }

    const int n = t.cols;

    double xlo = std::numeric_limits<double>::max();
    double xhi = std::numeric_limits<double>::lowest();
    double ylo = xlo;
    double yhi = xhi;
    int    valid = 0;

    for (int i = 0; i < n; ++i)
    {
        const double x = t.at<double>(0, i);
        const double y = t.at<double>(1, i);
        if (!Finite(x) || !Finite(y))
        {
            continue;
        }
        xlo = std::min(xlo, x); xhi = std::max(xhi, x);
        ylo = std::min(ylo, y); yhi = std::max(yhi, y);
        ++valid;
    }

    if (valid == 0)
    {
        return Finish(NewCanvas(st), st);
    }

    if (xhi - xlo < 1e-12) { xlo -= 0.5; xhi += 0.5; }
    if (yhi - ylo < 1e-12) { ylo -= 0.5; yhi += 0.5; }

    const double xp = (xhi - xlo) * st.y_pad_ratio;
    const double yp = (yhi - ylo) * st.y_pad_ratio;
    xlo -= xp; xhi += xp;
    ylo -= yp; yhi += yp;

    if (!st.auto_x && st.x_max > st.x_min) { xlo = st.x_min; xhi = st.x_max; }
    if (!st.auto_y && st.y_max > st.y_min) { ylo = st.y_min; yhi = st.y_max; }

    cv::Mat canvas = NewCanvas(st);
    Painter pt(canvas, std::max(1, st.supersample), st.text_renderer);

    const double rx = st.margin_left;
    const double ry = st.margin_top;
    const double rw = st.width - st.margin_left - st.margin_right;
    const double rh = st.height - st.margin_top - st.margin_bottom;

    DrawGridAndFrame(pt, st, rx, ry, rw, rh, xlo, xhi, ylo, yhi);

    LinearAxis ax;
    ax.lo = xlo; ax.hi = xhi; ax.p0 = rx; ax.p1 = rx + rw;

    LinearAxis ay;
    ay.lo = ylo; ay.hi = yhi; ay.p0 = ry + rh; ay.p1 = ry;

    const cv::Scalar color = ColorAt(0);

    for (int i = 0; i < n; ++i)
    {
        const double x = t.at<double>(0, i);
        const double y = t.at<double>(1, i);
        if (!Finite(x) || !Finite(y))
        {
            continue;
        }
        pt.Dot(ax.Map(x), ay.Map(y), st.marker_radius, color);
    }

    DrawTitle(pt, st, 0.5 * st.width);

    if (!st.legend.empty() && !st.legend[0].empty())
    {
        DrawLegend(pt, st, {st.legend[0]}, {color}, rx, ry, rw, rh);
    }

    DrawAxisLabels(pt, st, rx, ry, rw, rh);

    return Finish(canvas, st);
}

/* ========================================================================== */
/*                                  Bar chart                                 */
/* ========================================================================== */

cv::Mat PlotBar(const cv::Mat &values, const Style &st)
{
    cv::Mat m = ToF64(values);
    if (m.empty())
    {
        return Finish(NewCanvas(st), st);
    }

    std::vector<std::vector<double>> all = SplitSeries(m);
    if (all.empty())
    {
        return Finish(NewCanvas(st), st);
    }

    /* 1-D input -> one series, each bar forming its own group */
    const bool   single   = (all.size() == 1U);
    const size_t series_n = all.size();
    const size_t group_n  = all[0].size();

    if (group_n == 0U)
    {
        return Finish(NewCanvas(st), st);
    }

    /* ---- Value range (bar charts always include 0) ---- */
    double ylo = 0.0;
    double yhi = 0.0;
    MinMax(all, ylo, yhi);
    ylo = std::min(ylo, 0.0);
    yhi = std::max(yhi, 0.0);
    if (yhi - ylo < 1e-12) { yhi = ylo + 1.0; }

    const double pad = (yhi - ylo) * st.y_pad_ratio;
    if (yhi > 0.0) { yhi += pad; }
    if (ylo < 0.0) { ylo -= pad; }

    cv::Mat canvas = NewCanvas(st);
    Painter pt(canvas, std::max(1, st.supersample), st.text_renderer);

    const double rx = st.margin_left;
    const double ry = st.margin_top;
    const double rw = st.width - st.margin_left - st.margin_right;
    const double rh = st.height - st.margin_top - st.margin_bottom;

    LinearAxis ay;
    ay.lo = ylo; ay.hi = yhi; ay.p0 = ry + rh; ay.p1 = ry;

    /* ---- Grid + frame + y ticks (x is discrete by group, drawn by hand below) ---- */
    {
        const bool auto_ystep = (st.y_tick_step <= 0.0);
        double sy = 0.0;
        const std::vector<double> yt = auto_ystep ? Ticks(ylo, yhi, 6, sy) : std::vector<double>();
        const double ystep = auto_ystep ? sy : st.y_tick_step;

        if (st.grid)
        {
            if (auto_ystep)
            {
                for (double v : yt) { pt.Line(rx, ay.Map(v), rx + rw, ay.Map(v), st.grid_color, 1.0); }
            }
            else
            {
                for (double v = std::ceil(ylo / ystep) * ystep; v <= yhi; v += ystep)
                {
                    pt.Line(rx, ay.Map(v), rx + rw, ay.Map(v), st.grid_color, 1.0);
                }
            }
        }

        pt.Rect(rx, ry, rw, rh, st.frame_color, 1.2);

        const double tsc = st.font_scale * 0.82;
        auto one = [&](double v) {
            pt.Line(rx - st.tick_len, ay.Map(v), rx, ay.Map(v), st.foreground, 1.2);
            pt.Text(NumToStr(v, st.decimals), rx - st.tick_len - 4.0, ay.Map(v),
                    tsc, st.foreground, Align::Right, VAlign::Middle);
        };

        if (auto_ystep) { for (double v : yt) { one(v); } }
        else
        {
            for (double v = std::ceil(ylo / ystep) * ystep; v <= yhi; v += ystep) { one(v); }
        }
    }

    /* ---- Bars ---- */
    const double slot  = rw / static_cast<double>(group_n);
    const double inner = slot * st.bar_fill_ratio;
    const double bw    = single ? inner : inner / static_cast<double>(series_n);

    for (size_t g = 0; g < group_n; ++g)
    {
        const double gx = rx + slot * static_cast<double>(g) + 0.5 * slot;

        for (size_t s = 0; s < series_n; ++s)
        {
            if (g >= all[s].size())
            {
                continue;
            }

            const double v = all[s][g];
            if (!Finite(v))
            {
                continue;
            }

            const double x0 = single
                ? (gx - 0.5 * bw)
                : (gx - 0.5 * inner + bw * static_cast<double>(s));

            const double y0 = ay.Map(0.0);
            const double y1 = ay.Map(v);

            const cv::Scalar color = ColorAt(static_cast<int>(s));
            const double top = std::min(y0, y1);
            const double hgt = std::fabs(y1 - y0);

            pt.FillRect(x0, top, bw, hgt, color);
            pt.Rect(x0, top, bw, hgt, st.frame_color, 1.0);

            if (st.bar_value_labels)
            {
                pt.Text(NumToStr(v, st.decimals),
                        x0 + 0.5 * bw, (v >= 0.0 ? y1 - 4.0 : y1 + 4.0),
                        st.font_scale * 0.72, st.foreground, Align::Center,
                        (v >= 0.0) ? VAlign::Bottom : VAlign::Top);
            }
        }

        /* Group label: legend holds group names here */
        const std::string gl = (g < st.legend.size() && !st.legend[g].empty())
            ? st.legend[g]
            : NumToStr(static_cast<double>(g), 0);

        pt.Text(gl, gx, ry + rh + st.tick_len + 4.0, st.font_scale * 0.82,
                st.foreground, Align::Center, VAlign::Top);
    }

    DrawTitle(pt, st, 0.5 * st.width);

    /* Grouped bars get auto-generated series legends S1/S2... (legend holds group names) */
    if (!single)
    {
        std::vector<std::string> names;
        std::vector<cv::Scalar>  cols;
        for (size_t s = 0; s < series_n; ++s)
        {
            names.push_back("S" + std::to_string(s + 1));
            cols.push_back(ColorAt(static_cast<int>(s)));
        }
        DrawLegend(pt, st, names, cols, rx, ry, rw, rh);
    }

    DrawAxisLabels(pt, st, rx, ry, rw, rh);

    return Finish(canvas, st);
}

/* ========================================================================== */
/*                                 Radar chart                                */
/* ========================================================================== */

cv::Mat PlotRadar(const cv::Mat &values, const Style &st,
                  const std::vector<std::string> &axis_labels)
{
    cv::Mat m = ToF64(values);
    if (m.empty())
    {
        return Finish(NewCanvas(st), st);
    }

    std::vector<std::vector<double>> all = SplitSeries(m);
    if (all.empty())
    {
        return Finish(NewCanvas(st), st);
    }

    const size_t axes = all[0].size();
    if (axes < 3U)
    {
        return Finish(NewCanvas(st), st);   /* Fewer than 3 axes cannot form a polygon */
    }

    /* ---- Full scale per axis ---- */
    std::vector<double> scale(axes, 0.0);
    for (size_t a = 0; a < axes; ++a)
    {
        double mx = 0.0;
        for (const auto &s : all)
        {
            if (a < s.size() && Finite(s[a]))
            {
                mx = std::max(mx, s[a]);
            }
        }
        scale[a] = (st.radar_axis_max > 0.0) ? st.radar_axis_max : std::max(mx, 1e-9);
    }

    cv::Mat canvas = NewCanvas(st);
    Painter pt(canvas, std::max(1, st.supersample), st.text_renderer);

    const double cx = 0.5 * st.width;
    const double cy = 0.5 * st.height + st.margin_top * 0.25;

    /* Radius from min(w,h) keeps a true circle and leaves room for the axis labels */
    const double R = std::min(st.width, st.height) * 0.5 -
                     std::max(48.0, std::min(st.width, st.height) * 0.16);

    const double pi = 3.14159265358979323846;

    /* Axis 0 points up; the rest are distributed clockwise */
    auto angle = [&](size_t a) {
        return -pi * 0.5 + 2.0 * pi * static_cast<double>(a) / static_cast<double>(axes);
    };

    /* ---- Grid rings ---- */
    const int rings = std::max(1, st.radar_rings);

    for (int k = 1; k <= rings; ++k)
    {
        const double r = R * static_cast<double>(k) / static_cast<double>(rings);
        std::vector<cv::Point2d> ring;
        for (size_t a = 0; a < axes; ++a)
        {
            const double t = angle(a);
            ring.push_back(cv::Point2d(cx + r * std::cos(t), cy + r * std::sin(t)));
        }
        ring.push_back(ring.front());
        pt.Polyline(ring, st.grid_color, 1.1);
    }

    /* ---- Radial axes ---- */
    for (size_t a = 0; a < axes; ++a)
    {
        const double t = angle(a);
        pt.Line(cx, cy, cx + R * std::cos(t), cy + R * std::sin(t), st.grid_color, 1.1);
    }

    /* ---- Polygons of each series ---- */
    for (size_t s = 0; s < all.size(); ++s)
    {
        const cv::Scalar color = ColorAt(static_cast<int>(s));

        std::vector<cv::Point2d> poly;
        poly.reserve(axes + 1U);

        for (size_t a = 0; a < axes; ++a)
        {
            const double v = (a < all[s].size() && Finite(all[s][a])) ? all[s][a] : 0.0;
            double ratio = (scale[a] > 0.0) ? (v / scale[a]) : 0.0;
            ratio = std::max(0.0, std::min(1.0, ratio));

            const double t = angle(a);
            poly.push_back(cv::Point2d(cx + R * ratio * std::cos(t),
                                       cy + R * ratio * std::sin(t)));
        }

        if (poly.empty())
        {
            continue;
        }

        /* Fill (translucent) */
        if (st.radar_fill_alpha > 0.0)
        {
            cv::Mat overlay = canvas.clone();
            Painter op(overlay, std::max(1, st.supersample), st.text_renderer);
            op.FillPoly(poly, color);
            cv::addWeighted(overlay, st.radar_fill_alpha, canvas,
                            1.0 - st.radar_fill_alpha, 0.0, canvas);
        }

        /* Outline */
        std::vector<cv::Point2d> closed = poly;
        closed.push_back(poly.front());
        pt.Polyline(closed, color, st.line_width + 0.4);

        /* Vertex markers */
        for (const auto &p : poly)
        {
            pt.Dot(p.x, p.y, st.marker_radius, color);
        }
    }

    /* ---- Axis labels (alignment per quadrant so they never cover the chart) ---- */
    for (size_t a = 0; a < axes; ++a)
    {
        const double t  = angle(a);
        const double lx = cx + (R + 18.0) * std::cos(t);
        const double ly = cy + (R + 18.0) * std::sin(t);

        const std::string lbl = (a < axis_labels.size() && !axis_labels[a].empty())
            ? axis_labels[a]
            : ("A" + std::to_string(a + 1));

        Align  ha = Align::Center;
        VAlign va = VAlign::Middle;

        const double c = std::cos(t);
        const double sn = std::sin(t);
        if (c > 0.35)       { ha = Align::Left; }
        else if (c < -0.35) { ha = Align::Right; }
        if (sn > 0.35)      { va = VAlign::Top; }
        else if (sn < -0.35){ va = VAlign::Bottom; }

        pt.Text(lbl, lx, ly, st.font_scale * 0.88, st.foreground, ha, va);
    }

    DrawTitle(pt, st, 0.5 * st.width);

    /* ---- Legend (below the chart so it does not cover the radar) ---- */
    if (!st.legend.empty())
    {
        std::vector<std::string> names;
        std::vector<cv::Scalar>  cols;
        for (size_t s = 0; s < all.size(); ++s)
        {
            names.push_back(s < st.legend.size() ? st.legend[s] : std::string());
            cols.push_back(ColorAt(static_cast<int>(s)));
        }

        Style st2 = st;
        st2.legend_top_right = false;
        DrawLegend(pt, st2, names, cols,
                   st.margin_left, st.height - st.margin_bottom - 10.0,
                   st.width - st.margin_left - st.margin_right, 0.0);
    }

    return Finish(canvas, st);
}

/* ========================================================================== */
/*                              Pseudo-color map                              */
/* ========================================================================== */

cv::Mat PlotPseudoColor(const cv::Mat &data, const Style &st,
                        int colormap, bool colorbar)
{
    cv::Mat m = ToF64(data);
    if (m.empty())
    {
        return Finish(NewCanvas(st), st);
    }

    /* ---- Normalize (skip NaN) ---- */
    double lo = std::numeric_limits<double>::max();
    double hi = std::numeric_limits<double>::lowest();

    for (int r = 0; r < m.rows; ++r)
    {
        for (int c = 0; c < m.cols; ++c)
        {
            const double v = m.at<double>(r, c);
            if (!Finite(v))
            {
                continue;
            }
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
    }

    if (lo > hi)
    {
        lo = 0.0;
        hi = 1.0;
    }

    /* For a fixed color scale: auto_y = false + y_min/y_max (used as vmin/vmax here) */
    if (!st.auto_y && st.y_max > st.y_min)
    {
        lo = st.y_min;
        hi = st.y_max;
    }
    if (hi - lo < 1e-12)
    {
        hi = lo + 1.0;
    }

    cv::Mat idx(m.rows, m.cols, CV_8UC1);
    cv::Mat nanmask(m.rows, m.cols, CV_8UC1, cv::Scalar(0));

    for (int r = 0; r < m.rows; ++r)
    {
        for (int c = 0; c < m.cols; ++c)
        {
            const double v = m.at<double>(r, c);
            if (!Finite(v))
            {
                idx.at<uchar>(r, c)     = 0;
                nanmask.at<uchar>(r, c) = 255;
                continue;
            }
            double t = (v - lo) / (hi - lo);
            t = std::max(0.0, std::min(1.0, t));
            idx.at<uchar>(r, c) = static_cast<uchar>(cvRound(t * 255.0));
        }
    }

    cv::Mat color;
    cv::applyColorMap(idx, color, colormap);

    /* Paint NaN white so it is distinguishable from the minimum */
    color.setTo(cv::Scalar(255, 255, 255), nanmask);

    /* ---- Canvas and layout ---- */
    cv::Mat canvas = NewCanvas(st);
    Painter pt(canvas, std::max(1, st.supersample), st.text_renderer);

    const double barw      = (colorbar && st.colorbar_width > 0) ? st.colorbar_width : 0.0;
    const double bar_gap   = (barw > 0.0) ? 12.0 : 0.0;
    const double bar_ticks = (barw > 0.0) ? 62.0 : 0.0;   /* Width of the tick text right of the colorbar */

    const double rx = st.margin_left;
    const double ry = st.margin_top;
    const double rw = st.width - st.margin_left - st.margin_right - barw - bar_gap - bar_ticks;
    const double rh = st.height - st.margin_top - st.margin_bottom;

    if (rw < 40.0 || rh < 40.0)
    {
        return Finish(canvas, st);   /* Canvas too small for a usable plot area */
    }

    /* ---- Scale the color map into the plot area ---- */
    {
        cv::Mat scaled;
        cv::resize(color, scaled,
                   cv::Size(std::max(1, cvRound(rw * pt.Scale())),
                            std::max(1, cvRound(rh * pt.Scale()))),
                   0.0, 0.0, cv::INTER_NEAREST);

        const cv::Point tl = pt.P(rx, ry);
        scaled.copyTo(pt.Canvas()(cv::Rect(tl.x, tl.y, scaled.cols, scaled.rows)));

        /* Cell grid (only meaningful for small matrices) */
        if (st.heatmap_grid && m.rows <= 40 && m.cols <= 40)
        {
            for (int r = 1; r < m.rows; ++r)
            {
                const double y = ry + rh * static_cast<double>(r) / static_cast<double>(m.rows);
                pt.Line(rx, y, rx + rw, y, cv::Scalar(80, 80, 80), 1.0);
            }
            for (int c = 1; c < m.cols; ++c)
            {
                const double x = rx + rw * static_cast<double>(c) / static_cast<double>(m.cols);
                pt.Line(x, ry, x, ry + rh, cv::Scalar(80, 80, 80), 1.0);
            }
        }

        pt.Rect(rx, ry, rw, rh, st.frame_color, 1.2);
    }

    /* ---- Axis ticks (using matrix indices) ---- */
    {
        double sx = 0.0;
        double sy = 0.0;

        const std::vector<double> xt = Ticks(0.0, static_cast<double>(m.cols - 1), 6, sx);
        const std::vector<double> yt = Ticks(0.0, static_cast<double>(m.rows - 1), 6, sy);

        const double tsc = st.font_scale * 0.82;

        for (double v : xt)
        {
            const double px = rx + rw * v / std::max(1.0, static_cast<double>(m.cols - 1));
            pt.Line(px, ry + rh, px, ry + rh + st.tick_len, st.foreground, 1.2);
            pt.Text(NumToStr(v, 0), px, ry + rh + st.tick_len + 4.0,
                    tsc, st.foreground, Align::Center, VAlign::Top);
        }

        for (double v : yt)
        {
            const double py = ry + rh * v / std::max(1.0, static_cast<double>(m.rows - 1));
            pt.Line(rx - st.tick_len, py, rx, py, st.foreground, 1.2);
            pt.Text(NumToStr(v, 0), rx - st.tick_len - 4.0, py,
                    tsc, st.foreground, Align::Right, VAlign::Middle);
        }
    }

    /* ---- Colorbar ---- */
    if (barw > 0.0)
    {
        const double bx = rx + rw + bar_gap;

        /* Build the colorbar directly as a vertical 256-step ramp, with row 0 holding the
         * lowest value so low values stay at the bottom. Do **not** build a horizontal ramp
         * and call cv::rotate: rotate swaps width and height, so the following ROI size no
         * longer matches (barw, rh) and the Mat ROI assertion fires. */
        cv::Mat ramp(256, 1, CV_8UC1);
        for (int i = 0; i < 256; ++i)
        {
            ramp.at<uchar>(255 - i, 0) = static_cast<uchar>(i);
        }

        cv::Mat rampColor;
        cv::applyColorMap(ramp, rampColor, colormap);

        cv::Mat rampV;
        cv::resize(rampColor, rampV,
                   cv::Size(std::max(1, cvRound(barw * pt.Scale())),
                            std::max(1, cvRound(rh * pt.Scale()))),
                   0.0, 0.0, cv::INTER_NEAREST);

        const cv::Point tl = pt.P(bx, ry);
        rampV.copyTo(pt.Canvas()(cv::Rect(tl.x, tl.y, rampV.cols, rampV.rows)));

        pt.Rect(bx, ry, barw, rh, st.frame_color, 1.0);

        double sy = 0.0;
        const std::vector<double> ct = Ticks(lo, hi, 6, sy);
        const double tsc = st.font_scale * 0.78;

        for (double v : ct)
        {
            const double t  = (v - lo) / (hi - lo);
            const double py = ry + rh * (1.0 - t);
            pt.Line(bx + barw, py, bx + barw + 5.0, py, st.foreground, 1.2);
            pt.Text(NumToStr(v, st.decimals), bx + barw + 7.0, py,
                    tsc, st.foreground, Align::Left, VAlign::Middle);
        }
    }

    /* ---- Title and axis labels ---- */
    DrawTitle(pt, st, 0.5 * st.width);

    if (!st.xlabel.empty())
    {
        pt.Text(st.xlabel, rx + 0.5 * rw, st.height - 16.0,
                st.font_scale * 0.95, st.foreground, Align::Center, VAlign::Bottom);
    }
    if (!st.ylabel.empty())
    {
        /* The y label of a pseudo-color map is drawn horizontally above the plot area so it
         * never overlaps the rotated tick labels */
        pt.Text(st.ylabel, rx, ry - 12.0, st.font_scale * 0.95,
                st.foreground, Align::Center, VAlign::Bottom);
    }

    return Finish(canvas, st);
}

/* ========================================================================== */
/*                              Subplot tiling                                */
/* ========================================================================== */

cv::Mat TileFigures(const std::vector<cv::Mat> &figs,
                    int cols,
                    int pad,
                    const std::vector<std::string> &labels,
                    cv::Scalar bg)
{
    if (figs.empty())
    {
        return cv::Mat();
    }

    const int n = static_cast<int>(figs.size());
    if (cols <= 0)
    {
        cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n))));
    }
    cols = std::max(1, std::min(cols, n));
    const int rows = (n + cols - 1) / cols;

    int cell_w = 0;
    int cell_h = 0;
    for (const auto &f : figs)
    {
        cell_w = std::max(cell_w, f.cols);
        cell_h = std::max(cell_h, f.rows);
    }

    if (cell_w <= 0 || cell_h <= 0)
    {
        return cv::Mat();
    }

    cv::Mat out(rows * cell_h + (rows + 1) * pad,
                cols * cell_w + (cols + 1) * pad, CV_8UC3, bg);

    for (int i = 0; i < n; ++i)
    {
        const int r = i / cols;
        const int c = i % cols;

        cv::Mat fig = figs[static_cast<size_t>(i)];
        if (fig.empty())
        {
            continue;
        }

        /* Scale proportionally into the cell */
        const double s = std::min(static_cast<double>(cell_w) / fig.cols,
                                  static_cast<double>(cell_h) / fig.rows);
        cv::Mat scaled;
        if (std::fabs(s - 1.0) > 1e-3)
        {
            cv::resize(fig, scaled, cv::Size(), s, s, cv::INTER_AREA);
        }
        else
        {
            scaled = fig;
        }

        const int ox = pad + c * (cell_w + pad) + (cell_w - scaled.cols) / 2;
        const int oy = pad + r * (cell_h + pad) + (cell_h - scaled.rows) / 2;

        scaled.copyTo(out(cv::Rect(ox, oy, scaled.cols, scaled.rows)));

        if (i < static_cast<int>(labels.size()) && !labels[static_cast<size_t>(i)].empty())
        {
            cv::putText(out, labels[static_cast<size_t>(i)],
                        cv::Point(ox + 6, oy + 22), cv::FONT_HERSHEY_SIMPLEX,
                        0.7, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        }
    }

    return out;
}

/* ========================================================================== */
/*                               Saving figures                               */
/* ========================================================================== */

bool Save(const cv::Mat &img, const std::string &path)
{
    if (img.empty() || path.empty())
    {
        return false;
    }

    try
    {
        return cv::imwrite(path, img);
    }
    catch (const cv::Exception &)
    {
        return false;
    }
}

}   // namespace plot
}   // namespace lidar
