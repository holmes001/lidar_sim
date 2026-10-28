/**
 * @file    plot.h
 * @brief   基于 OpenCV 的论文插图绘制库（**声明**；实现见 Src/plot.cpp）
 *
 * @details
 * 本模块提供一组**直接输出 cv::Mat 图像**的绘图函数，用于生成论文/报告里常见的
 * 统计图形。设计目标：
 *
 *  1. **不依赖 highgui**：所有函数只往内存里的 `cv::Mat` 上画，不创建窗口，
 *     因此可以在服务器 / 批处理 / CI 里跑。
 *  2. **论文级抗锯齿**：内部按 `Style::supersample` 倍超采样绘制，最后用
 *     `cv::INTER_AREA` 降采样。OpenCV 的 `cv::line` 本身没有抗锯齿，
 *     这是让曲线不出现台阶的最有效手段（默认 3 倍）。
 *  3. **输入统一为一维/二维数组**：所有接口都接受 `cv::Mat`（内部自动转换到
 *     `CV_64F`），也提供 `Make1D()` / `Make2D()` 方便从 `std::vector` 构造。
 *     一维表示单条序列，二维表示多条序列（默认**每行一条序列**）。
 *  4. **声明与实现分离**：本头文件只包含声明，实现在 `Src/plot.cpp`。
 *     头文件仅依赖 `opencv2/core.hpp` 与 `opencv2/imgproc.hpp`（后者只为
 *     `cv::COLORMAP_*` 的默认参数），**不拉 imgcodecs / highgui / videoio**，
 *     所以包含它的编译单元开销很小。
 *
 * @section supported_charts 支持的图形
 *
 * | 函数 | 图形 | 论文里的典型用途 |
 * |---|---|---|
 * | `PlotLine()`        | 折线图 | 时域波形、收敛曲线、多方法对比 |
 * | `PlotLineBand()`    | 均值±标准差带折线 | 多次重复实验的统计结果（阴影带） |
 * | `PlotScatter()`     | 散点图 | 实测值 vs 理论值、相关性 |
 * | `PlotBar()`         | 柱状图（可分组） | 各方案指标对比、消融实验 |
 * | `PlotRadar()`       | 雷达图（多组多边形） | 多指标综合评价 |
 * | `PlotPseudoColor()` | 伪彩色图（含色条） | 二维场分布（光强、误差、代价函数） |
 * | `TileFigures()`     | 多子图拼接 | 组合成 (a)(b)(c) 子图 |
 *
 * @section usage 典型用法
 *
 * @code
 * #include "plot.h"
 *
 * int main()
 * {
 *     using namespace lidar::plot;
 *
 *     // ---- 折线图：两条曲线 ----
 *     std::vector<std::vector<double>> data = {
 *         {0.0, 0.8, 0.9, 0.95, 0.99},     // 第 1 条
 *         {0.0, 0.5, 0.7, 0.80, 0.85}      // 第 2 条
 *     };
 *     Style st;
 *     st.title  = "Convergence";
 *     st.xlabel = "Iteration";
 *     st.ylabel = "Residual";
 *     st.legend = {"Proposed", "Baseline"};
 *     cv::Mat fig = PlotLine(Make2D(data), st);
 *     Save(fig, "convergence.png");
 *
 *     // ---- 伪彩色图：二维场 ----
 *     cv::Mat field = ...;                       // CV_64F，M×N
 *     Style st2;
 *     st2.title  = "Intensity";
 *     st2.xlabel = "x [pixel]";
 *     Save(PlotPseudoColor(field, st2, cv::COLORMAP_TURBO, true), "field.png");
 *     return 0;
 * }
 * @endcode
 *
 * @warning **中文标签的限制**
 * `cv::putText()` 用的是 Hershey 矢量字体，**只支持 ASCII 可见字符**，中文（以及
 * `±` / `σ` 这类非 ASCII 符号）会渲染成 `???`。如果插图需要中文标注：
 *   1. 标题/坐标轴标签用英文（论文正文本来就常用英文，推荐）；
 *   2. 给 `Style::text_renderer` 传一个自定义渲染回调（例如用 FreeType 加载
 *      `simhei.ttf` / `SourceHanSansSC.otf` 绘制），本库会把**所有**文字绘制
 *      都转交给该回调。注意回调负责按 `font_scale` 自行缩放字号。
 *
 * @warning 本库假定输入数据里 **NaN / Inf 表示"缺失点"**：折线会在该点断开
 *          （不会连一条穿越过去的直线），统计类函数会跳过该点，
 *          伪彩色图把它画成白色。
 *
 * @note 构建：把 `Src/plot.cpp` 加入编译目标即可。若使用本仓库的 CMakeLists，
 *       其中的 `GLOB_RECURSE` 会自动把 `Src` 下所有 cpp 源文件包含进来。
 *
 * @note 授权：随工程使用，无附加限制。
 */

#ifndef LIDAR_PLOT_H
#define LIDAR_PLOT_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>   /* 仅为 cv::COLORMAP_* 的默认参数 */

#include <functional>
#include <string>
#include <vector>

namespace lidar
{
namespace plot
{

/* ========================================================================== */
/*                                  枚举与别名                                 */
/* ========================================================================== */

/**
 * @brief 水平对齐方式（用于文字/图例定位）
 */
enum class Align
{
    Left = 0,   /**< 文字左边缘对齐到给定 x */
    Center,     /**< 文字水平居中于给定 x */
    Right       /**< 文字右边缘对齐到给定 x */
};

/**
 * @brief 垂直对齐方式（用于文字/图例定位）
 */
enum class VAlign
{
    Top = 0,    /**< 文字顶部对齐到给定 y */
    Middle,     /**< 文字垂直居中于给定 y */
    Bottom      /**< 文字基线/底部对齐到给定 y */
};

/**
 * @brief 自定义文字渲染回调
 * @param img        目标图像（内部是已超采样的画布，坐标已是像素）
 * @param text       要绘制的文本（UTF-8）
 * @param org        **左下角基线**起点（与 cv::putText 的 org 语义一致）
 * @param font_scale 字号缩放系数（已按超采样倍数放大）
 * @param color      颜色（BGR）
 * @param thickness  线宽（已按超采样倍数放大）
 * @note 传入 nullptr（默认）时使用 `cv::putText` + `FONT_HERSHEY_SIMPLEX`。
 * @see 文件头的"中文标签的限制"一节。
 */
using TextRenderer = std::function<void(cv::Mat &img,
                                        const std::string &text,
                                        cv::Point org,
                                        double font_scale,
                                        cv::Scalar color,
                                        int thickness)>;

/* ========================================================================== */
/*                                  样式配置                                   */
/* ========================================================================== */

/**
 * @brief 图形样式与标注
 *
 * @details 所有字段都有合理默认值，**只需要改你关心的那几个**。
 * 一次调用后不会被修改（绘图函数都按 const 引用接收），可以放心复用同一个 Style。
 *
 * @note 尺寸相关的字段（`width`/`height`/`margin_*`）都是**逻辑像素**；
 *       内部会乘以 `supersample` 得到真实画布大小，输出图像始终是
 *       `width × height`。
 */
struct Style
{
    /* ---- 画布与抗锯齿 ---- */

    /** @brief 输出图像宽度（逻辑像素） */
    int width  = 900;

    /** @brief 输出图像高度（逻辑像素） */
    int height = 600;

    /**
     * @brief 超采样倍数：内部按 width*ss × height*ss 绘制，最后降采样
     * @note 1 = 关闭（快，但有台阶）；3 = 论文级（默认）；4~6 = 极致，
     *       代价是内存 O(ss²)。
     */
    int supersample = 3;

    /* ---- 颜色（OpenCV 是 BGR 顺序！） ---- */

    /** @brief 背景色，默认白 */
    cv::Scalar background{255, 255, 255};

    /** @brief 前景色（坐标轴、刻度文字），默认黑 */
    cv::Scalar foreground{0, 0, 0};

    /** @brief 网格线颜色，默认浅灰 */
    cv::Scalar grid_color{220, 220, 220};

    /** @brief 绘图区边框颜色，默认黑 */
    cv::Scalar frame_color{0, 0, 0};

    /* ---- 边距（逻辑像素） ---- */

    int margin_left   = 92;   /**< 左边距，放 y 轴刻度与标签 */
    int margin_right  = 32;   /**< 右边距（伪彩色图会被自动留出空间给色条） */
    int margin_top    = 56;   /**< 上边距，放标题 */
    int margin_bottom = 74;   /**< 下边距，放 x 轴刻度与标签 */

    /* ---- 字体与线宽 ---- */

    /** @brief 字号缩放（相对 Hershey 基准），刻度字会再乘 0.82 */
    double font_scale = 0.62;

    /** @brief 线条粗细基准（逻辑像素） */
    int thickness = 2;

    /** @brief 曲线线宽（逻辑像素，支持小数） */
    double line_width = 2.4;

    /** @brief 刻度线长度（逻辑像素） */
    int tick_len = 7;

    /* ---- 内容标注 ---- */

    std::string title;      /**< 标题（空 = 不画） */
    std::string xlabel;     /**< x 轴标签（空 = 不画） */
    std::string ylabel;     /**< y 轴标签（空 = 不画；折线图会自动旋转 90°） */

    /** @brief 图例文字，按序列顺序；空字符串项会被跳过 */
    std::vector<std::string> legend;

    /** @brief 是否画网格 */
    bool grid = true;

    /** @brief 图例位置：true = 绘图区右上角，false = 绘图区下方 */
    bool legend_top_right = true;

    /* ---- 数值轴控制 ---- */

    /** @brief x 轴下界（仅 `auto_x == false` 时生效） */
    double x_min = 0.0;

    /** @brief x 轴上界（仅 `auto_x == false` 时生效） */
    double x_max = 0.0;

    /** @brief true = 忽略 x_min/x_max，按数据自动定界 */
    bool auto_x = true;

    /** @brief true = 按数据自动定界 y 轴 */
    bool auto_y = true;

    /** @brief y 轴下界（仅 `auto_y == false` 时生效；伪彩色图用作 vmin） */
    double y_min = 0.0;

    /** @brief y 轴上界（仅 `auto_y == false` 时生效；伪彩色图用作 vmax） */
    double y_max = 0.0;

    /** @brief y 轴是否强制包含 0（柱状图内部默认包含） */
    bool y_include_zero = false;

    /** @brief 自动定界时上下各留的余量比例（0.08 = 8%） */
    double y_pad_ratio = 0.08;

    /** @brief 手动指定 x 刻度间隔；<= 0 表示按 "1-2-5" 规则自动选取 */
    double x_tick_step = 0.0;

    /** @brief 手动指定 y 刻度间隔；<= 0 表示按 "1-2-5" 规则自动选取 */
    double y_tick_step = 0.0;

    /** @brief 刻度显示的小数位数；< 0 表示用 "%.4g" 短格式自动处理 */
    int decimals = -1;

    /* ---- 散点 / 柱状图 ---- */

    /** @brief 散点标记半径（逻辑像素） */
    int marker_radius = 4;

    /** @brief 是否在折线图的每个数据点画标记 */
    bool show_markers = false;

    /** @brief 柱状图柱子宽度占组内空位的比例 (0, 1] */
    double bar_fill_ratio = 0.8;

    /** @brief 柱状图是否在柱顶标数值 */
    bool bar_value_labels = false;

    /* ---- 雷达图 ---- */

    /**
     * @brief 雷达图各轴的统一最大值；<= 0 表示**每个轴各自**按该轴最大值归一化
     * @note 若你的指标已经归一化到 [0,1]，直接留 0 即可。
     */
    double radar_axis_max = 0.0;

    /** @brief 雷达图网格环数 */
    int radar_rings = 5;

    /** @brief 雷达图多边形填充透明度 [0,1]，0 = 不填充 */
    double radar_fill_alpha = 0.25;

    /* ---- 统计带 ---- */

    /** @brief 均值±标准差带的透明度 [0,1] */
    double band_alpha = 0.32;

    /* ---- 伪彩色图 ---- */

    /** @brief 伪彩色色条宽度（逻辑像素）；0 = 不画色条 */
    int colorbar_width = 26;

    /** @brief 伪彩色图是否在格子上画细网格（仅当行列数 <= 40 时生效） */
    bool heatmap_grid = false;

    /* ---- 扩展 ---- */

    /** @brief 自定义文字渲染器；nullptr = 用 cv::putText（仅 ASCII） */
    TextRenderer text_renderer = nullptr;
};

/* ========================================================================== */
/*                              输入数据便捷构造                               */
/* ========================================================================== */

/**
 * @brief 把一维 `std::vector` 转成绘图接口接受的列向量 `cv::Mat`（N×1, CV_64F）
 * @param v 数据
 * @return N×1 的 `CV_64F` 矩阵
 */
cv::Mat Make1D(const std::vector<double> &v);

/**
 * @brief 把二维 `std::vector` 转成 `cv::Mat`（rows×cols, CV_64F）
 * @param v 二维数据，要求每行长度一致
 * @return rows×cols 的 `CV_64F` 矩阵
 * @throws std::invalid_argument 各行长度不一致时
 */
cv::Mat Make2D(const std::vector<std::vector<double>> &v);

/* ========================================================================== */
/*                                  折线图                                     */
/* ========================================================================== */

/**
 * @brief 折线图（支持多条曲线、自动刻度、图例、网格）
 *
 * @param series          一维或二维数组（内部自动转 `CV_64F`）
 *                        - **一维**（N×1 或 1×N）：一条曲线
 *                        - **二维**（M×N）：**每行一条曲线**（M 条、每条 N 点）
 * @param st              样式与标注
 * @param series_in_cols  true 时把二维输入的**每列**当作一条曲线
 * @return BGR 图像（尺寸 = `st.width × st.height`）
 *
 * @details
 * - 横坐标默认取采样序号 0, 1, …, N-1；需要自定义横坐标时请用 @ref PlotScatter。
 * - **NaN / Inf 表示缺失点**：曲线在该处断开，不会连出穿过去的直线。
 * - y 轴自动定界时按 `st.y_pad_ratio` 上下留白；`st.y_include_zero` 可强制包含 0。
 * - 刻度按 "1-2-5 × 10ⁿ" 规则选取，可用 `st.x_tick_step` / `st.y_tick_step` 覆盖。
 *
 * @par 示例
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
/*                            均值 ± 标准差带折线                              */
/* ========================================================================== */

/**
 * @brief 折线图 + 半透明误差带（`mean ± k·σ`）
 *
 * @param samples 二维数组，**每行是一次独立观测**，每列是同一采样位置
 *                （即 `rows = 重复实验次数`，`cols = 采样点数`）
 * @param st      样式与标注
 * @param k_sigma 误差带宽度，单位标准差（论文常用 1σ 或 2σ）
 * @return BGR 图像
 *
 * @details 这是论文里表达"多次重复实验结果"的标准画法：实线画均值，
 *          阴影带画 ±kσ。带用 `st.band_alpha` 控制透明度，先画带再画线，
 *          保证曲线不被遮住。
 *
 * @note 每列的 NaN 会被**跳过**（不计入该列的均值和方差）。
 *       若某列有效样本为 0，该处曲线断开。
 * @note `st.legend[0]` 为空时自动生成图例名 `mean +/- k sigma`（纯 ASCII，
 *       避免 `±` / `σ` 被 cv::putText 渲染成 `??`）。
 *
 * @par 示例
 * @code
 * cv::Mat samples(20, 100, CV_64F);   // 20 次实验 × 100 个采样点
 * Style st;
 * st.title = "Position Error (mean +/- 1 sigma, N=20)";
 * Save(PlotLineBand(samples, st, 1.0), "band.png");
 * @endcode
 */
cv::Mat PlotLineBand(const cv::Mat &samples, const Style &st = Style(), double k_sigma = 1.0);

/* ========================================================================== */
/*                          带显式横坐标的折线图                               */
/* ========================================================================== */

/**
 * @brief 折线图（**显式给定横坐标**，用于频率轴 / 距离轴等物理横轴）
 *
 * @param x              横坐标，一维数组（N 或 N×1），长度须等于采样点数
 * @param series         纵坐标，一维或二维数组
 *                       - **一维**（N）：一条曲线
 *                       - **二维**（M×N）：每行一条曲线
 * @param st             样式与标注
 * @param series_in_cols true 时把二维输入的每列当作一条曲线
 * @return BGR 图像
 *
 * @details 与 @ref PlotLine 完全相同，唯一区别是**横坐标由调用方给定**而不是
 *          采样序号 0..N−1。凡是横轴有物理含义（频率、时间、距离、角度）的图
 *          都应该用它，否则刻度数字会和真实物理量对不上。
 *
 *          自动定界时 x 轴按 `min(x)`~`max(x)` 取，并按 "1-2-5" 规则选刻度；
 *          若 `st.auto_x == false`，仍以 `st.x_min` / `st.x_max` 为准。
 *
 * @note `x` 的长度与采样点数不一致时返回空白图（避免越界）。
 * @note `x` 允许非等间隔（例如对数频率轴），绘图时按给定顺序直接连线。
 *
 * @par 示例：画多普勒谱（横轴是真实频率）
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
/*                                  散点图                                     */
/* ========================================================================== */

/**
 * @brief 散点图（x-y 数据对）
 *
 * @param xy             二维数组，形状 **N×2**（每行一个点 `(x, y)`）或 **2×N**
 * @param st             样式与标注
 * @param points_in_rows true 表示 N×2（默认）；false 表示 2×N
 * @return BGR 图像
 *
 * @details 与 @ref PlotLine 的区别：这里的**横坐标是真实数据**，而不是采样序号。
 *          适合画"实测 vs 理论"、"相关性"这类图。若 `st.legend[0]` 非空，
 *          会作为散点系列的图例名。
 *
 * @note NaN 点会被跳过；坐标轴按有效点的范围自动定界并留白。
 */
cv::Mat PlotScatter(const cv::Mat &xy, const Style &st = Style(), bool points_in_rows = true);

/* ========================================================================== */
/*                                  柱状图                                     */
/* ========================================================================== */

/**
 * @brief 柱状图（单系列或分组对比）
 *
 * @param values 一维或二维数组
 *               - **一维**（N）：N 根柱子，每根自成一"组"
 *               - **二维**（M×N）：**每行一组**，每组 N 根柱子并排（M 个系列）
 * @param st     样式与标注
 * @return BGR 图像
 *
 * @details
 * - y 轴自动包含 0（柱状图从 0 起才有可比性）。
 * - 柱子宽度由 `st.bar_fill_ratio` 控制；分组时组内并排。
 * - `st.bar_value_labels = true` 可在柱顶标数值（矩阵较大时拥挤，慎用）。
 * - **x 轴刻度标签取 `st.legend` 里对应项**（组名）；为空则显示组序号。
 * - 分组柱状图的系列图例自动生成为 `S1`/`S2`…，因为 `legend` 已被组名占用。
 *
 * @note 支持负值（柱子向下生长）。
 */
cv::Mat PlotBar(const cv::Mat &values, const Style &st = Style());

/* ========================================================================== */
/*                                  雷达图                                     */
/* ========================================================================== */

/**
 * @brief 雷达图 / 蜘蛛网图（多指标综合评价）
 *
 * @param values      一维或二维数组
 *                    - **一维**（N）：一个多边形，N 个轴
 *                    - **二维**（M×N）：M 个多边形，每个 N 个轴（每行一组）
 * @param st          样式与标注
 * @param axis_labels 各轴名称，长度应为 N；空或长度不符时用 `A1`/`A2`… 代替
 * @return BGR 图像
 *
 * @details
 * - 轴从正上方（12 点方向）开始，**顺时针**均匀分布。
 * - 每轴归一化：默认以该轴在所有序列中的最大值为满量程（`radar_axis_max <= 0`）；
 *   若希望所有轴共用同一量程（指标同量纲时），给 `st.radar_axis_max` 赋值。
 * - 网格环数由 `st.radar_rings` 控制；填充透明度 `st.radar_fill_alpha`。
 * - 轴标签画在多边形外侧，按象限自动选择对齐方向，避免压住图形。
 *
 * @note 雷达半径按 `min(width, height)` 计算以保持正圆，雷达居中；
 *       输出尺寸仍是 `st.width × st.height`。
 * @note 轴数少于 3 时返回空白图（无法构成多边形）。
 *
 * @par 示例
 * @code
 * std::vector<std::vector<double>> v = {
 *     {0.92, 0.85, 0.78, 0.95, 0.70},   // 方案 A
 *     {0.75, 0.80, 0.88, 0.70, 0.85}    // 方案 B
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
/*                                 伪彩色图                                    */
/* ========================================================================== */

/**
 * @brief 伪彩色图（热力图）+ 可选色条
 *
 * @param data      二维数组（M×N），例如二维场分布、误差矩阵、代价函数
 * @param st        样式与标注
 * @param colormap  颜色映射，建议 `cv::COLORMAP_JET` / `COLORMAP_TURBO` /
 *                  `COLORMAP_VIRIDIS` / `COLORMAP_HOT` / `COLORMAP_BONE`
 * @param colorbar  是否在右侧画色条（含数值刻度）
 * @return BGR 图像
 *
 * @details
 * - 数据先按最小/最大值线性归一化到 0..255，再套 `cv::applyColorMap`。
 * - **NaN 会被画成白色**（便于区分"无数据"和"最小值"），不影响归一化范围。
 * - 色条刻度用与数据相同的 `%.4g` / `st.decimals` 格式，保证和正文一致。
 * - 行索引 0 在**图像顶部**（与矩阵书写顺序一致）。
 * - `st.heatmap_grid = true` 时在格子边界画细线（行列数 <= 40 才画）。
 * - 需要固定色标量程时，置 `st.auto_y = false` 并给 `st.y_min` / `st.y_max`
 *   赋值（它们在这里被当作 vmin/vmax）。
 *
 * @par 示例
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
/*                                多子图拼接                                   */
/* ========================================================================== */

/**
 * @brief 把多张图拼成 (a)(b)(c) 形式的子图网格
 *
 * @param figs   各子图（尺寸可以不同，会被等比缩放到格子里并居中）
 * @param cols   每行列数（<= 0 时自动取 `ceil(sqrt(n))`）
 * @param pad    格子间距（像素）
 * @param labels 子图编号，例如 `{"(a)", "(b)", "(c)"}`；空则不标
 * @param bg     拼图背景色
 * @return 拼接后的图像
 *
 * @details 论文里常见"一张图里 2×2 子图"的需求。本函数等分格子、等比缩放
 *          每张子图并居中，可选在左上角标注 `(a)`。
 *
 * @note 每个子图的**宽高比会被保持**（不拉伸）。
 * @note 编号文字用 `cv::putText` + `LINE_AA` 绘制，仅支持 ASCII。
 */
cv::Mat TileFigures(const std::vector<cv::Mat> &figs,
                    int cols = 0,
                    int pad = 12,
                    const std::vector<std::string> &labels = {},
                    cv::Scalar bg = cv::Scalar(255, 255, 255));

/* ========================================================================== */
/*                                  保存图片                                   */
/* ========================================================================== */

/**
 * @brief 把图像写盘
 * @param img  要保存的图像（BGR）
 * @param path 文件名，扩展名决定格式（`.png` / `.jpg` / `.bmp` / `.tif` / `.pdf` 等）
 * @return true 成功；false 失败（`img` 为空、目录不存在、格式不支持等）
 *
 * @note **论文插图建议用 PNG**：无损、体积可控，避免 JPEG 在曲线边缘产生振铃。
 * @note 本函数需要 `opencv_imgcodecs` 模块。若某个目标只画图不写盘，可以不链接它。
 */
bool Save(const cv::Mat &img, const std::string &path);

}   // namespace plot
}   // namespace lidar

#endif /* LIDAR_PLOT_H */
