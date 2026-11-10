/**
 * @file    plot.h
 * @brief   OpenCV-based publication figure library (**declarations**; see Src/plot.cpp)
 *
 * @details
 * This module provides plotting functions that draw **directly into a cv::Mat**,
 * producing the statistical figures typically found in papers and reports. Goals:
 *
 *  1. **No highgui dependency**: every function draws into an in-memory `cv::Mat`
 *     and never opens a window, so it runs on servers, in batch jobs and in CI.
 *  2. **Publication-grade anti-aliasing**: drawing happens at `Style::supersample`
 *     times the target size and is then downsampled with `cv::INTER_AREA`.
 *     OpenCV's `cv::line` has no anti-aliasing, so this is the most effective way
 *     to keep curves free of stair steps (default 3x).
 *  3. **Uniform 1-D / 2-D input**: every entry point accepts `cv::Mat` (converted
 *     internally to `CV_64F`), and `Make1D()` / `Make2D()` build one from a
 *     `std::vector`. 1-D means one series, 2-D means several series (by default
 *     **one series per row**).
 *  4. **Declaration/implementation split**: this header declares only; the
 *     implementation is in `Src/plot.cpp`. The header depends only on
 *     `opencv2/core.hpp` and `opencv2/imgproc.hpp` (the latter only for the
 *     `cv::COLORMAP_*` defaults) and does **not** pull in imgcodecs / highgui /
 *     videoio, so translation units including it stay cheap.
 *
 * @section supported_charts Supported charts
 *
 * | Function | Chart | Typical use in a paper |
 * |---|---|---|
 * | `PlotLine()`        | Line plot | Time-domain waveforms, convergence curves, comparisons |
 * | `PlotLineBand()`    | Mean +/- std band | Statistics over repeated experiments (shaded band) |
 * | `PlotScatter()`     | Scatter plot | Measured vs theoretical values, correlation |
 * | `PlotBar()`         | Bar chart (grouped) | Metric comparison across schemes, ablation |
 * | `PlotRadar()`       | Radar chart (multiple polygons) | Multi-metric evaluation |
 * | `PlotPseudoColor()` | Pseudo-color map (with colorbar) | 2-D fields (intensity, error, cost) |
 * | `TileFigures()`     | Subplot tiling | Combining (a)(b)(c) subfigures |
 *
 * @section usage Typical usage
 *
 * @code
 * #include "plot.h"
 *
 * int main()
 * {
 *     using namespace lidar::plot;
 *
 *     // ---- Line plot: two curves ----
 *     std::vector<std::vector<double>> data = {
 *         {0.0, 0.8, 0.9, 0.95, 0.99},     // series 1
 *         {0.0, 0.5, 0.7, 0.80, 0.85}      // series 2
 *     };
 *     Style st;
 *     st.title  = "Convergence";
 *     st.xlabel = "Iteration";
 *     st.ylabel = "Residual";
 *     st.legend = {"Proposed", "Baseline"};
 *     cv::Mat fig = PlotLine(Make2D(data), st);
 *     Save(fig, "convergence.png");
 *
 *     // ---- Pseudo-color map: 2-D field ----
 *     cv::Mat field = ...;                       // CV_64F, M x N
 *     Style st2;
 *     st2.title  = "Intensity";
 *     st2.xlabel = "x [pixel]";
 *     Save(PlotPseudoColor(field, st2, cv::COLORMAP_TURBO, true), "field.png");
 *     return 0;
 * }
 * @endcode
 *
 * @warning **Non-ASCII label limitation**
 * `cv::putText()` uses the Hershey vector fonts, which support **only visible
 * ASCII**; CJK characters (and non-ASCII symbols) render as `???`. If a figure
 * needs CJK labels:
 *   1. Use English for titles and axis labels (recommended; papers usually do);
 *   2. Pass a custom render callback to `Style::text_renderer` (e.g. draw with
 *      FreeType and a CJK font such as `SourceHanSansSC.otf`). The library routes
 *      **all** text drawing through that callback, which must scale the font size
 *      by `font_scale` itself.
 *
 * @warning This library assumes **NaN / Inf marks a missing point**: lines break
 *          there (no straight segment across the gap), statistical functions skip
 *          such points, and pseudo-color maps paint them white.
 *
 * @note Build: add `Src/plot.cpp` to the compile target. With this repository's
 *       CMakeLists, `GLOB_RECURSE` pulls in every cpp under `Src` automatically.
 *
 * @note License: free to use within this project, no additional restrictions.
 */

#ifndef LIDAR_PLOT_H
#define LIDAR_PLOT_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>   /* only for the cv::COLORMAP_* defaults */

#include <functional>
#include <string>
#include <vector>

namespace lidar
{
namespace plot
{

/* ========================================================================== */
/*                              Enums and aliases                             */
/* ========================================================================== */

/**
 * @brief Horizontal alignment (for text/legend placement)
 */
enum class Align
{
    Left = 0,   /**< Left edge of the text aligns to the given x */
    Center,     /**< Text is horizontally centered on the given x */
    Right       /**< Right edge of the text aligns to the given x */
};

/**
 * @brief Vertical alignment (for text/legend placement)
 */
enum class VAlign
{
    Top = 0,    /**< Top of the text aligns to the given y */
    Middle,     /**< Text is vertically centered on the given y */
    Bottom      /**< Text baseline/bottom aligns to the given y */
};

/**
 * @brief Custom text rendering callback
 * @param img        Target image (already supersampled; coordinates are pixels)
 * @param text       Text to draw (UTF-8)
 * @param org        **Bottom-left baseline** origin (same meaning as cv::putText)
 * @param font_scale Font scale (already multiplied by the supersample factor)
 * @param color      Color (BGR)
 * @param thickness  Line thickness (already multiplied by the supersample factor)
 * @note Passing nullptr (default) uses `cv::putText` + `FONT_HERSHEY_SIMPLEX`.
 * @see The "Non-ASCII label limitation" section at the top of this file.
 */
using TextRenderer = std::function<void(cv::Mat &img,
                                        const std::string &text,
                                        cv::Point org,
                                        double font_scale,
                                        cv::Scalar color,
                                        int thickness)>;

/* ========================================================================== */
/*                                  Style                                     */
/* ========================================================================== */

/**
 * @brief Figure style and annotations
 *
 * @details Every field has a sensible default, so **only change what matters to you**.
 * A call never modifies it (drawing functions take it by const reference), so the
 * same Style can be reused freely.
 *
 * @note Size fields (`width`/`height`/`margin_*`) are **logical pixels**; they are
 *       multiplied by `supersample` internally, and the output image is always
 *       `width x height`.
 */
struct Style
{
    /* ---- Canvas and anti-aliasing ---- */

    /** @brief Output image width (logical pixels) */
    int width  = 900;

    /** @brief Output image height (logical pixels) */
    int height = 600;

    /**
     * @brief Supersample factor: draw at width*ss x height*ss, then downsample
     * @note 1 = off (fast but stair-stepped); 3 = publication grade (default);
     *       4-6 = extreme, at a memory cost of O(ss^2).
     */
    int supersample = 3;

    /* ---- Colors (OpenCV uses BGR order!) ---- */

    /** @brief Background color, white by default */
    cv::Scalar background{255, 255, 255};

    /** @brief Foreground color (axes, tick labels), black by default */
    cv::Scalar foreground{0, 0, 0};

    /** @brief Grid line color, light gray by default */
    cv::Scalar grid_color{220, 220, 220};

    /** @brief Plot frame color, black by default */
    cv::Scalar frame_color{0, 0, 0};

    /* ---- Margins (logical pixels) ---- */

    int margin_left   = 92;   /**< Left margin, holds y ticks and label */
    int margin_right  = 32;   /**< Right margin (pseudo-color maps reserve space for the colorbar) */
    int margin_top    = 56;   /**< Top margin, holds the title */
    int margin_bottom = 74;   /**< Bottom margin, holds x ticks and label */

    /* ---- Font and line width ---- */

    /** @brief Font scale (relative to the Hershey baseline); tick labels use 0.82x */
    double font_scale = 0.62;

    /** @brief Base line thickness (logical pixels) */
    int thickness = 2;

    /** @brief Curve line width (logical pixels, fractional allowed) */
    double line_width = 2.4;

    /** @brief Tick length (logical pixels) */
    int tick_len = 7;

    /* ---- Annotations ---- */

    std::string title;      /**< Title (empty = not drawn) */
    std::string xlabel;     /**< x axis label (empty = not drawn) */
    std::string ylabel;     /**< y axis label (empty = not drawn; auto-rotated 90 deg for line plots) */

    /** @brief Legend entries in series order; empty strings are skipped */
    std::vector<std::string> legend;

    /** @brief Whether to draw the grid */
    bool grid = true;

    /** @brief Legend position: true = top-right inside the plot, false = below it */
    bool legend_top_right = true;

    /* ---- Numeric axis control ---- */

    /** @brief x lower bound (used only when `auto_x == false`) */
    double x_min = 0.0;

    /** @brief x upper bound (used only when `auto_x == false`) */
    double x_max = 0.0;

    /** @brief true = ignore x_min/x_max and fit the data */
    bool auto_x = true;

    /** @brief true = fit the y axis to the data */
    bool auto_y = true;

    /** @brief y lower bound (used only when `auto_y == false`; vmin for pseudo-color) */
    double y_min = 0.0;

    /** @brief y upper bound (used only when `auto_y == false`; vmax for pseudo-color) */
    double y_max = 0.0;

    /** @brief Force the y axis to include 0 (bar charts always do) */
    bool y_include_zero = false;

    /** @brief Padding ratio added at both ends when fitting (0.08 = 8%) */
    double y_pad_ratio = 0.08;

    /** @brief Manual x tick step; <= 0 selects one by the "1-2-5" rule */
    double x_tick_step = 0.0;

    /** @brief Manual y tick step; <= 0 selects one by the "1-2-5" rule */
    double y_tick_step = 0.0;

    /** @brief Decimals shown on ticks; < 0 uses the short "%.4g" format */
    int decimals = -1;

    /* ---- Scatter / bar chart ---- */

    /** @brief Scatter marker radius (logical pixels) */
    int marker_radius = 4;

    /** @brief Draw a marker at every data point of a line plot */
    bool show_markers = false;

    /** @brief Bar width as a fraction of its slot (0, 1] */
    double bar_fill_ratio = 0.8;

    /** @brief Print the value on top of each bar */
    bool bar_value_labels = false;

    /* ---- Radar chart ---- */

    /**
     * @brief Common maximum for all radar axes; <= 0 normalizes **each axis** by its
     *        own maximum
     * @note If your metrics are already normalized to [0,1], leave it at 0.
     */
    double radar_axis_max = 0.0;

    /** @brief Number of radar grid rings */
    int radar_rings = 5;

    /** @brief Radar polygon fill alpha [0,1]; 0 = no fill */
    double radar_fill_alpha = 0.25;

    /* ---- Statistical band ---- */

    /** @brief Alpha of the mean +/- std band [0,1] */
    double band_alpha = 0.32;

    /* ---- Pseudo-color map ---- */

    /** @brief Colorbar width (logical pixels); 0 = no colorbar */
    int colorbar_width = 26;

    /** @brief Draw thin cell borders (only when rows and cols are <= 40) */
    bool heatmap_grid = false;

    /* ---- Extension ---- */

    /** @brief Custom text renderer; nullptr = cv::putText (ASCII only) */
    TextRenderer text_renderer = nullptr;
};

/* ========================================================================== */
/*                          Input data helpers                                */
/* ========================================================================== */

/**
 * @brief Convert a 1-D `std::vector` into a column-vector `cv::Mat` (N x 1, CV_64F)
 * @param v Data
 * @return N x 1 `CV_64F` matrix
 */
cv::Mat Make1D(const std::vector<double> &v);

/**
 * @brief Convert a 2-D `std::vector` into a `cv::Mat` (rows x cols, CV_64F)
 * @param v 2-D data; every row must have the same length
 * @return rows x cols `CV_64F` matrix
 * @throws std::invalid_argument If the rows have different lengths
 */
cv::Mat Make2D(const std::vector<std::vector<double>> &v);

/* ========================================================================== */
/*                                 Line plot                                  */
/* ========================================================================== */

/**
 * @brief Line plot (multiple curves, automatic ticks, legend, grid)
 *
 * @param series          1-D or 2-D array (converted internally to `CV_64F`)
 *                        - **1-D** (N x 1 or 1 x N): one curve
 *                        - **2-D** (M x N): **one curve per row** (M curves of N points)
 * @param st              Style and annotations
 * @param series_in_cols  If true, treat **each column** of a 2-D input as one curve
 * @return BGR image (size = `st.width x st.height`)
 *
 * @details
 * - x defaults to the sample index 0, 1, ..., N-1; use @ref PlotLineXY for a custom x.
 * - **NaN / Inf marks a missing point**: the curve breaks there instead of drawing
 *   a straight segment across the gap.
 * - When y is fitted automatically, `st.y_pad_ratio` adds padding on both ends;
 *   `st.y_include_zero` forces 0 into range.
 * - Ticks follow the "1-2-5 x 10^n" rule; override with `st.x_tick_step` /
 *   `st.y_tick_step`.
 *
 * @par Example
 * @code
 * Style st;
 * st.title  = "Trajectory Error";
 * st.xlabel = "Frame";
 * st.ylabel = "Error [mm]";
 * st.legend = {"X", "Y"};
 * Save(PlotLine(two_by_n, st), "err.png");
 * @endcode
 */
cv::Mat PlotLine(const cv::Mat &series, const Style &st = Style(), bool series_in_cols = false);

/* ========================================================================== */
/*                        Mean +/- std band line plot                         */
/* ========================================================================== */

/**
 * @brief Line plot with a translucent error band (`mean +/- k*sigma`)
 *
 * @param samples 2-D array, **one row per independent observation**, one column
 *                per sample position (i.e. `rows = repeats`, `cols = samples`)
 * @param st      Style and annotations
 * @param k_sigma Band half-width in standard deviations (1 or 2 typically)
 * @return BGR image
 *
 * @details The standard way to show results of repeated experiments: a solid line
 *          for the mean and a shaded band for +/-k sigma. `st.band_alpha` sets the
 *          band transparency; the band is drawn before the line so the curve stays
 *          visible.
 *
 * @note NaN in a column is **skipped** (excluded from that column's mean and
 *       variance). A column with no valid samples leaves a break in the curve.
 * @note When `st.legend[0]` is empty, the legend entry is generated as
 *       `mean +/- k sigma` (pure ASCII, so cv::putText does not render `??`).
 *
 * @par Example
 * @code
 * cv::Mat samples(20, 100, CV_64F);   // 20 runs x 100 samples
 * Style st;
 * st.title = "Position Error (mean +/- 1 sigma, N=20)";
 * Save(PlotLineBand(samples, st, 1.0), "band.png");
 * @endcode
 */
cv::Mat PlotLineBand(const cv::Mat &samples, const Style &st = Style(), double k_sigma = 1.0);

/* ========================================================================== */
/*                      Line plot with explicit x axis                        */
/* ========================================================================== */

/**
 * @brief Line plot with an **explicitly given x axis** (for frequency, range, etc.)
 *
 * @param x              x axis, 1-D array (N or N x 1), length must match the samples
 * @param series         y values, 1-D or 2-D array
 *                       - **1-D** (N): one curve
 *                       - **2-D** (M x N): one curve per row
 * @param st             Style and annotations
 * @param series_in_cols If true, treat each column of a 2-D input as one curve
 * @return BGR image
 *
 * @details Identical to @ref PlotLine except that **x is supplied by the caller**
 *          instead of the sample index 0..N-1. Use it whenever the horizontal axis
 *          has physical meaning (frequency, time, range, angle), otherwise the tick
 *          numbers will not match the real quantity.
 *
 *          When fitting automatically, x spans `min(x)`..`max(x)` and ticks follow
 *          the "1-2-5" rule; when `st.auto_x == false`, `st.x_min` / `st.x_max` win.
 *
 * @note If the length of `x` does not match the number of samples, a blank image is
 *       returned (prevents out-of-bounds access).
 * @note `x` may be non-uniformly spaced (e.g. a logarithmic frequency axis); points
 *       are connected in the given order.
 *
 * @par Example: plotting a Doppler spectrum (x is the real frequency)
 * @code
 * Style st;
 * st.xlabel = "Frequency (Hz)";
 * st.ylabel = "Magnitude";
 * Save(PlotLineXY(Make1D(spec.frequency_hz), Make1D(spec.magnitude), st),
 *      "spectrum.png");
 * @endcode
 */
cv::Mat PlotLineXY(const cv::Mat &x, const cv::Mat &series,
                   const Style &st = Style(), bool series_in_cols = false);

/* ========================================================================== */
/*                                Scatter plot                                */
/* ========================================================================== */

/**
 * @brief Scatter plot (x-y data pairs)
 *
 * @param xy             2-D array shaped **N x 2** (one `(x, y)` per row) or **2 x N**
 * @param st             Style and annotations
 * @param points_in_rows true = N x 2 (default); false = 2 x N
 * @return BGR image
 *
 * @details Unlike @ref PlotLine, **x here is real data** rather than a sample index.
 *          Suited to "measured vs theoretical" or correlation plots. If
 *          `st.legend[0]` is non-empty it names the scatter series.
 *
 * @note NaN points are skipped; axes are fitted to the valid points with padding.
 */
cv::Mat PlotScatter(const cv::Mat &xy, const Style &st = Style(), bool points_in_rows = true);

/* ========================================================================== */
/*                                  Bar chart                                 */
/* ========================================================================== */

/**
 * @brief Bar chart (single series or grouped comparison)
 *
 * @param values 1-D or 2-D array
 *               - **1-D** (N): N bars, each forming its own group
 *               - **2-D** (M x N): **one group per row**, N bars side by side (M series)
 * @param st     Style and annotations
 * @return BGR image
 *
 * @details
 * - The y axis always includes 0 (bars must start at 0 to be comparable).
 * - Bar width is set by `st.bar_fill_ratio`; grouped bars sit side by side.
 * - `st.bar_value_labels = true` prints values above the bars (crowded for large
 *   matrices, use with care).
 * - **x tick labels come from `st.legend`** (group names); empty entries fall back
 *   to the group index.
 * - Grouped charts generate series legend entries `S1`/`S2`... because `legend` is
 *   already used for group names.
 *
 * @note Negative values are supported (bars grow downwards).
 */
cv::Mat PlotBar(const cv::Mat &values, const Style &st = Style());

/* ========================================================================== */
/*                                 Radar chart                                */
/* ========================================================================== */

/**
 * @brief Radar / spider chart (multi-metric evaluation)
 *
 * @param values      1-D or 2-D array
 *                    - **1-D** (N): one polygon with N axes
 *                    - **2-D** (M x N): M polygons of N axes (one per row)
 * @param st          Style and annotations
 * @param axis_labels Axis names, expected length N; empty or mismatched entries
 *                    fall back to `A1`/`A2`...
 * @return BGR image
 *
 * @details
 * - Axes start at 12 o'clock and are distributed **clockwise**.
 * - Per-axis normalization: by default each axis uses its own maximum across all
 *   series (`radar_axis_max <= 0`); set `st.radar_axis_max` for a common scale.
 * - Grid ring count comes from `st.radar_rings`; fill alpha from `st.radar_fill_alpha`.
 * - Axis labels are drawn outside the polygon, with alignment chosen per quadrant so
 *   they never overlap the chart.
 *
 * @note The radius is derived from `min(width, height)` to keep a true circle and the
 *       chart is centered; the output size is still `st.width x st.height`.
 * @note Fewer than 3 axes returns a blank image (no polygon possible).
 *
 * @par Example
 * @code
 * std::vector<std::vector<double>> v = {
 *     {0.92, 0.85, 0.78, 0.95, 0.70},   // method A
 *     {0.75, 0.80, 0.88, 0.70, 0.85}    // method B
 * };
 * Style st;
 * st.title  = "Performance";
 * st.legend = {"Method A", "Method B"};
 * Save(PlotRadar(Make2D(v), st, {"Acc", "Speed", "Robust", "Cost", "Power"}), "radar.png");
 * @endcode
 */
cv::Mat PlotRadar(const cv::Mat &values, const Style &st = Style(),
                  const std::vector<std::string> &axis_labels = {});

/* ========================================================================== */
/*                              Pseudo-color map                              */
/* ========================================================================== */

/**
 * @brief Pseudo-color (heat) map with an optional colorbar
 *
 * @param data     2-D array (M x N), e.g. a 2-D field, error matrix or cost function
 * @param st       Style and annotations
 * @param colormap Color map, e.g. `cv::COLORMAP_JET` / `COLORMAP_TURBO` /
 *                 `COLORMAP_VIRIDIS` / `COLORMAP_HOT` / `COLORMAP_BONE`
 * @param colorbar Whether to draw a colorbar (with numeric ticks) on the right
 * @return BGR image
 *
 * @details
 * - Data is normalized linearly to 0..255 by its min/max, then passed to
 *   `cv::applyColorMap`.
 * - **NaN is painted white** (distinguishes "no data" from the minimum) and does not
 *   affect the normalization range.
 * - Colorbar ticks use the same `%.4g` / `st.decimals` format as the data.
 * - Row index 0 is at the **top** of the image (matching matrix order).
 * - `st.heatmap_grid = true` draws thin cell borders (only when rows and cols <= 40).
 * - For a fixed color scale, set `st.auto_y = false` and assign `st.y_min` /
 *   `st.y_max` (used here as vmin/vmax).
 *
 * @par Example
 * @code
 * cv::Mat field(64, 64, CV_64F);
 * Style st;
 * st.title  = "Far-field Intensity";
 * st.xlabel = "x [pixel]";
 * Save(PlotPseudoColor(field, st, cv::COLORMAP_TURBO, true), "field.png");
 * @endcode
 */
cv::Mat PlotPseudoColor(const cv::Mat &data, const Style &st = Style(),
                        int colormap = cv::COLORMAP_JET, bool colorbar = true);

/* ========================================================================== */
/*                              Subplot tiling                                */
/* ========================================================================== */

/**
 * @brief Tile several figures into an (a)(b)(c) subplot grid
 *
 * @param figs   Subfigures (sizes may differ; each is scaled proportionally and centered)
 * @param cols   Columns per row (<= 0 picks `ceil(sqrt(n))` automatically)
 * @param pad    Spacing between cells (pixels)
 * @param labels Subfigure tags such as `{"(a)", "(b)", "(c)"}`; empty = none
 * @param bg     Background color of the tiled image
 * @return Tiled image
 *
 * @details Papers often need a 2x2 subplot figure. This function splits the canvas
 *          into equal cells, scales each subfigure into its cell, centers it, and
 *          optionally tags it `(a)` in the top-left corner.
 *
 * @note The **aspect ratio of each subfigure is preserved** (no stretching).
 * @note Tags are drawn with `cv::putText` + `LINE_AA`, so ASCII only.
 */
cv::Mat TileFigures(const std::vector<cv::Mat> &figs,
                    int cols = 0,
                    int pad = 12,
                    const std::vector<std::string> &labels = {},
                    cv::Scalar bg = cv::Scalar(255, 255, 255));

/* ========================================================================== */
/*                               Saving figures                               */
/* ========================================================================== */

/**
 * @brief Write an image to disk
 * @param img  Image to save (BGR)
 * @param path File name; the extension selects the format (`.png` / `.jpg` / `.bmp` /
 *             `.tif` / `.pdf` ...)
 * @return true on success; false on failure (empty `img`, missing directory,
 *         unsupported format ...)
 *
 * @note **Prefer PNG for paper figures**: lossless and compact, avoiding the ringing
 *       JPEG produces around curve edges.
 * @note This function requires the `opencv_imgcodecs` module. Targets that only draw
 *       figures and never write them need not link it.
 */
bool Save(const cv::Mat &img, const std::string &path);

}   // namespace plot
}   // namespace lidar

#endif /* LIDAR_PLOT_H */
