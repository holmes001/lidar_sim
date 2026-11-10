# lidar_sim {#mainpage}

脉冲多普勒雷达/激光雷达链路的最小可读实现：发射复包络 → 目标回波 → 功率闭环 → 匹配滤波测距 → 慢时间 FFT 测速，每一步的中间结果都落盘到 `figs/`。

本文件同时充当 Doxygen 主页（`USE_MDFILE_AS_MAINPAGE = README.md`），标题层级会自动变成章节，`\warning` / `\note` 会渲染成告示框。

\tableofcontents

## 数据流 {#dataflow}

```
generate_a_signal(u)                       发射复包络 u，采样率 dt（可选过采样）
      │
      ├──► make_reference(u)               切一个符号作为匹配滤波模板
      │
recieve_signal(u, radar, object, wave, t_offset, echo_path)
      │       分数延迟重采样 + 多普勒相位 + 雷达方程衰减 + 高斯噪声
      ▼
amplifyToTargetPower(rx, target)           闭环把平均功率拉到目标值
      │
matchedFilter(rx, ref)                     快时间压缩 → 距离
      │
      ▼
recieve_frame(...) → matchedFilterFrame(...) → rangeDopplerMap(...)
      │                                        慢时间 FFT → 距离-多普勒图
      ▼
findRangeDopplerPeak(...)                  峰值 → 距离 / 多普勒 / 速度
```

**两个维度必须分清**，否则后面几乎每个"奇怪现象"都解释不通。

| 维度 | 索引 | 对应的物理量 | 产生的环节 |
|---|---|---|---|
| 快时间 | 采样点 `i`、距离门 `r` | $R = r\,c\,dt/2$ | `generate_a_signal`、`matchedFilter` |
| 慢时间 | 脉冲号 `p`、多普勒 bin `k` | $v = f_d\lambda/2$ | `t_offset`、`rangeDopplerMap` |

## 模块职责 {#modules}

| 文件 | 关键符号 | 职责 |
|---|---|---|
| `Inc/typedef.h` | `Complex`、`ComplexVec`、`ComplexVec2D`、`kpi`、`c` | 全局类型与物理常数，**唯一**定义处 |
| `Inc/transport.h` | `radar`、`object`、`waveform`、`echo_path`、`recieve_signal`、`recieve_frame` | 雷达方程、波形、回波合成 |
| `Inc/filter.h` | `matchedFilter`、`MatchFilterMode` | 匹配滤波（脉冲压缩） |
| `Inc/doppler.h` | `matchedFilterFrame`、`rangeDopplerMap`、`findRangeDopplerPeak` | 距离-多普勒处理 |
| `Inc/power.h` | `PIController`、`amplifyToTargetPower` | 功率测量与增益闭环 |
| `Inc/random_error.h` | `addGaussianNoiseStd` | 复高斯噪声注入 |
| `Inc/app.h` | `fft`、`fftShift`、`time_axis`、`save_figure` | 模型无关的通用工具 |
| `Inc/plot.h` / `Src/plot.cpp` | `lidar::plot::Style`、`PlotLineXY`、`PlotPseudoColor` | 抗锯齿出版级绘图，直接画进 `cv::Mat` |
| `Src/main.cpp` | — | 演示编排：一次跑通全链路并出图 |

## transport：回波合成 {#transport}

### 双路径：`TxSource::Record` 与 `TxSource::Analytic`

`recieve_signal` 的第五个参数 `t_offset` 之后是一个 `echo_path`，决定包络从哪来：

| `source` | 包络取值 | 用途 |
|---|---|---|
| `Record`（默认） | 对**实际发射记录** `u` 做分数延迟重采样 | 主路径。发射端注入的误差（相位噪声、IQ 不平衡、功放非线性、量化）会自然进入回波 |
| `Analytic` | 用 `wave` 在连续时间重新求值 `wf_eval` | 参考实现，用于和 Record 对拍。**结构上不可能反映任何发射误差** |

\warning 记录模式只读 `u`，`wave` 参数被忽略（`WF_IMPULSE` 除外，见下）。如果哪天加了 `apply_tx_errors(u)`，只有 `Record` 模式才看得到效果——这是判断"改动是否真的生效"的最快方法。

\warning 记录模式会把超出 `u` 末尾的滞后**截断为 0**；Analytic 模式永远不会截断。两者在窗口边缘必然不一致，这是设计如此。

### 分数延迟：`lag = i - n_tau`，不是 `i*dt - tau`

```cpp
const double n_tau = round_trip_delay(object_1.distance) / radar_1.dt;  // 以为单位的延迟
const double lag   = static_cast<double>(i) - n_tau;                    // 先做无量纲减法
```

- 若写成 `i*dt - tau`，两项都在 $2\times10^{-5}\,\mathrm{s}$ 量级，相减会损失两位有效数字，导致边界样本与发射端的 `t = i*dt` 判断不一致（脉冲边沿差一个样本）。
- `lag` 同时喂给两条路径：`Analytic` 用它算 `t_local = lag*dt`，`Record` 用它算 `u` 的下标 `lag * u_oversample`。
- `t_offset` **只进入载波相位**，不进入包络。混进包络会让第 2 个脉冲起 `t` 落在波形窗口之外，回波恒为零。

### 冲激波形强制最近邻

`WF_IMPULSE` 无条件把 `path.kernel` 覆盖为 `DelayKernel::Nearest`：线性/三次核会把一个冲激样本劈成两个半幅样本，直接破坏"每周期至多一个非零样本"的前提，匹配滤波峰值也会摊平。

\warning 这是唯一一处"忽略调用方参数"的地方，改插值代码时不要把它当成冗余分支删掉。

### 插值核的选择依据

| 核 | 精度特征 | 适用 |
|---|---|---|
| `Nearest` | 无振铃，延迟误差最大 | 冲激串（强制） |
| `Linear` | 边沿处只摊到 1 个样本，**不振铃** | 矩形脉冲、方波等含跳变的包络 |
| `Cubic`（默认） | 平滑包络误差最小，但在跳变处振铃 4 个样本 | FMCW、单频 CW 等连续包络 |

实测（$dt$ 网格，0.5 样本延迟，无噪声）：`Linear` 与 `Analytic` 的相对 L2 误差 0.05，`Cubic` 0.0512。误差全部集中在矩形脉冲的**一个边沿样本**上——任何采样域插值都无法表示分数位置的无限带宽跳变，这是记录路径的物理代价，不是 bug。

### 过采样的作用（`u_oversample`）

`generate_a_signal(u, radar, wave, os)` 让 `u` 以 `dt/os` 采样，`recieve_signal` 用 `echo_path::u_oversample = os` 告知同一倍率。此时分数延迟的**小数部分被缩小 os 倍**，用最便宜的 `Linear` 就能达到 `Cubic` 的精度：$os=8$ 时上例误差降为 0。

\warning `u_oversample` 必须与生成 `u` 时用的倍率一致，且 `u` 的长度应为 `samp*os`（否则尾部被截断）。

### 匹配滤波参考必须与信号同一采样率

回波 `v` 永远输出在 `dt` 网格上（长度 `samp`），所以 `matchedFilter` 的模板也必须在这个网格上。用 `make_reference(a_os, ..., os)` 生成 8 倍长的模板去匹配，相关系数在 `[20102-799, 20003]` 区间内完全相同，`argmax` 会返回平台左端（19303 而不是 20003）。

\warning 峰值索引还会随模板长度整体平移，因为 `Same` 模式的索引语义是 `out[t] = full[t + M - 1]`（见下节）。跨不同长度的模板比较峰值索引没有意义。

### 载波相位 $\varphi_0$ 与 `fmod`

$\varphi_0 = 2kR = 4\pi R/\lambda$ 是往返传播带来的常数相位，多普勒只是它的时间导数，所以常数项必须单独保留（相干积累、干涉测距、多目标相位关系都要它）。

`fmod(..., 2π)` 折叠的目的是保精度：$R$ 大时 $4\pi R/\lambda$ 会到 $10^6$ 量级，double 在此量级的可分辨间隔迅速吃掉有效位，而相位本质只定义到 $2\pi$ 模。

### 噪声与随机种子

`recieve_signal` 每次调用都执行 `g_seed++`，因此连续调用得到不相关噪声、而重跑同一程序结果可复现。

\warning 这也意味着**插入任何一次 `recieve_signal` 调用都会改变后续所有噪声实现**。`main.cpp` 里的路径对比因此放在最后执行。

## filter：匹配滤波 {#filter}

- 实现是**直接时域卷积**，$y = x * h,\ h[k] = \overline{ref[M-1-k]}$，复杂度 $O(NM)$，不走 FFT。整条链路里唯一的 FFT 在 `rangeDopplerMap`。
- 输入全零时走 `continue` 稀疏快速路径，所以长稀疏记录（脉冲雷达）不会白算。
- `Same` 模式取 `full[M-1 : M-1+N]`，朴素但关键的语义：**`out[t]` 表示"从样本 t 开始匹配"的得分**。因此回波峰值索引直接就是回波起始样本，不需要减 `M-1`。
- `normalizeByRefEnergy = true` 时按模板总能量归一，完美匹配峰值为 1，便于跨模板比较。

## doppler：距离-多普勒 {#doppler}

### 矩阵方向

`matchedFilterFrame` 返回 `slow[p][r]`：**行 = 脉冲（慢时间），列 = 距离门（快时间）**。`rangeDopplerMap` 沿**行**做 FFT，即每个距离门做一次 $N$ 点 DFT：

$$X[k][r]=\sum_{p=0}^{N-1} s[p][r]\,e^{-j2\pi kp/N}$$

不要误以为"再对每个匹配滤波输出做一次 FFT"——那是对一整行做频谱分析，既测不出速度也丢掉了脉冲间相位。

### 列主序重排 + 一个 batched plan

`slow` 是 `vector<vector<...>>`，行内连续；直接按列取元素跨度 384 KB，全部掉缓存。所以先重排到 `buf[r*n_pulse + p]`，再：

| `fftw_plan_many_dft` 参数 | 值 | 含义 |
|---|---|---|
| `rank` / `n` | 1 / `&n_pulse` | 一维、长度 $N$ |
| `howmany` | `n_range` | **每个距离门一个独立变换** |
| `istride` / `idist` | 1 / `n_pulse` | 变换内连续，变换间间隔 $N$ |
| `sign` | `FFTW_FORWARD` | 核取 $e^{-j2\pi kp/N}$ |

\warning FFTW 也支持跨步输入（`istride = n_range`）免去重排，但那是为局部性做的取舍：一次 $O(NR)$ 拷贝的成本远小于 FFT 本身。

### fftShift 的行序

`rd[k][r]` 取原始 bin `(k + N/2) % N`。$N=64$ 时的行序：

| 行 | 0 | 1…31 | **32** | 33…63 |
|---|---|---|---|---|
| 频率 | $\pm f_s/2$（Nyquist，正负混叠同一点） | $-31df$…$-df$ | **0 Hz** | $+df$…$+31df$ |

`dopplerAxis()` 的 `(i - N/2)*df` 与此一致（第 0 行取 $-N/2\cdot df$ 而实际是 $+N/2\cdot df$，二者是同一混叠频率，不是 bug）。`findRangeDopplerPeak` 用 `bin = k0 + delta - N/2` 换算，三者自洽。

### 亚 bin 峰值

峰值先用整数网格搜索 $|X|^2$，再沿多普勒方向做三点抛物线插值；峰值落在首/末行时退化为整数行号。

\warning 抛物线插值只在**主瓣内**有意义。若峰值落在旁瓣或噪声尖峰上，插值结果会偏出半个 bin 以上，判据要自己加。

## power：功率闭环 {#power}

- 控制器是**位置式**离散 PI：`u = kp*e + integral`，误差与输出都归一化，因此 `kp/ki` 与信号单位无关。
- 积分分离：`|e| > integ_gate` 时只用 P 项；输出饱和且误差同向时冻结积分（抗饱和）。
- 归一化误差是 `e = (P_target - P_meas) / max(P_meas, P_target*1e-12)`（分母做除零保护），因此 `1 + u` 就是所需的功率比。功率正比于 $g^2$，所以增益修正要开方：`g *= sqrt(max(0, 1+u))`，再被 `max_gain` 夹住。这是回波幅度与功率混用最容易出错的一处。
- `PowerSenseMode::Analytic` 用 `P_out = g^2*P_in` 直接算，每轮 $O(1)$；`Measured` 每轮重建信号再测，$O(N)$，留给非标量模型（频响、饱和、AM-PM）。

\warning `PowerLoopConfig::history` 是指向外部容器的**裸指针**，必须保证其生命周期覆盖整个闭环调用；同时 `PIController::integral` 是跨调用状态，换对象前要 `reset()`。

## random_error：噪声 {#noise}

- `addGaussianNoiseStd` 的 `stddev` 是**每个 I/Q 分量**的标准差，I/Q 独立采样因此圆周对称，噪声功率与样本数无关（这正是弃用旧 API `addGaussianNoise` 的原因：它按总能量归一，改 `samp` 就改了实际 SNR）。
- `stddev == 0 && mean == 0` 时直接返回副本，所以把 `noise_sigma` 置零可做无噪声对照，且不消耗随机数。

## app / plot：绘图 {#plot}

- `plot` 库只往 `cv::Mat` 里画，不依赖 highgui，可在无显示环境跑。
- 抗锯齿靠**超采样**：先按 `Style::supersample`（默认 3×）尺寸绘制，再用 `cv::INTER_AREA` 缩回。`cv::line` 本身没有抗锯齿。
- `PlotPseudoColor` 复用 `Style::y_min/y_max` 作为色标 `vmin/vmax`（需把 `auto_y` 置 false），不是纵轴范围。
- 伪彩图的 y 轴标签是**水平居中绘制**、不旋转，长标签会被裁切，所以标签写 "doppler bin"、单位放进标题。
- 中文字符会渲染成 `???`：`cv::putText` 的 Hershey 字体只支持可见 ASCII。需要中文就换英文，或给 `Style::text_renderer` 传 FreeType 回调。
- `NaN/Inf` 表示缺失点：折线在此断开，统计函数跳过，伪彩图涂白。

## main：演示编排 {#main}

- 时间域图必须裁剪：发射窗只有 1 µs，而 $R=3000\,\mathrm{m}$ 的回波在 20 µs，整段 24 µs 完全看不清，所以用 `crop_window` 取 0–2 µs 与 19–22 µs。
- 伪彩图只取峰值距离门附近的窗口（±120 门）：24000 列挤进一张图就剩一条线。
- 伪彩图用相对全局峰值的 dB（下限 −120 dB）：线性标度下 sinc 旁瓣和噪底全黑，只剩一个亮点。
- 路径对比块放在 `main` 最后：见上文 `g_seed` 的说明。

## 数值陷阱清单 {#pitfalls}

| 现象 | 原因 | 处置 |
|---|---|---|
| 改了延迟路径却毫无变化 | $R=3000,\ dt=1\,\mathrm{ns} \Rightarrow n_\tau=20000$ **整数**样本，插值核误差恒为 0 | 用 `R = 3000.375`（$n_\tau = 20002.5$）做分数延迟测试 |
| $\varphi_0$ 删掉也没差别 | 同上参数下 $2R/\lambda = 200000$ 为整数，$\varphi_0 \equiv 0 \pmod{2\pi}$ | 换 $\lambda=0.031$ 或非整数 $2R/\lambda$ |
| 多普勒恒为 0 | 跨脉冲循环里 `t_offset` 一直传 0 | 传 `p*PRI`（`recieve_frame` 已处理） |
| 峰值索引比预期小几百 | 模板比回波长，`Same` 模式的相关峰退化成一个平台，`argmax` 返回平台左端；索引本身仍等于"匹配起点" | 模板与信号必须同一采样率、同一符号长度 |
| 相关系数出现平台 | 同上，平台宽度 = $M - M_{\text{echo}}$ | 用 `make_reference` 时传入正确的采样率 |
| `recieve_frame` 抛异常 | 接收窗 `samp*dt` 超过 `pri`，第 p 个窗会覆盖第 p+1 个回波 | 满足 `pri >= samp*dt` |
| 距离门与实际不符 | 幅值/索引用了错的那一维（快/慢时间混淆） | 见"数据流"一节的两维对照表 |

## 构建与运行 {#build}

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
./build/MyProject.exe        # Windows: .\build\MyProject.exe
```

依赖：OpenCV（`find_package`）、FFTW3（`pkg_check_modules`）、C++17。

\warning Ninja 有时会报 `no work to do` 而漏掉刚改的文件；touch 一下源文件时间戳再构建即可。

运行后所有图保存在 `figs/`，控制台输出匹配滤波峰值、距离-多普勒峰值、相邻脉冲相位步进，以及两条回波路径的交叉对比。

## 生成文档 {#doxygen}

```ini
PROJECT_NAME           = "lidar_sim"
OUTPUT_DIRECTORY       = docs
INPUT                  = Inc Src README.md
RECURSIVE              = YES
FILE_PATTERNS          = *.h *.cpp *.md
USE_MDFILE_AS_MAINPAGE = README.md
MARKDOWN_SUPPORT       = YES
EXTRACT_ALL            = NO
EXTRACT_STATIC         = YES
WARN_IF_UNDOCUMENTED   = YES
GENERATE_LATEX         = NO
```

本文件顶部的 `{#mainpage}` 让 Doxygen 把它当作主页面；正文的 `##` / `###` 自动成为 `\section` / `\subsection`，`\tableofcontents` 生成目录。代码侧的 `\defgroup` + `\ingroup`（例如把 `decay_factor` / `delay` / `generate_a_signal` 归入 "transport" 组）可以进一步把散装函数聚成模块页。

## 附：阅读代码时的疑问 {#faq}

**Q1 为什么 `recieve_signal(a, radar_1, object_1, wave_1)` 少传 `t_offset` 也不报错？**
默认实参写在 `Inc/transport.h` 的声明里（`double t_offset = 0.0`），编译器在调用点补上 `0.0`。默认实参不属于函数类型，也不能在 `.cpp` 的定义里重复写。`t_offset` 是"记录起始的绝对时刻"，跨脉冲时必须手动传 `p*PRI`，否则多普勒相位每个脉冲都从头开始，测速恒为 0。

**Q2 为什么 $f_d = 2v/\lambda$，而不是 $1+2v/c$？**
$2v/c$ 是**相对比值**（$f_r/f_0 \approx 1+2v/c$），$f_d$ 是**绝对频移**，二者相差一个 $f_0$：$f_d = f_0\cdot 2v/c = 2v/\lambda$。分子上的 2 来自往返两程。

**Q3 $\varphi_0 = 2kR = 4\pi R/\lambda$ 怎么来的？**
往返延迟 $\tau = 2R/c$，载波相位被推迟 $2\pi f_0\tau = 4\pi R/\lambda = 2kR$（$k = 2\pi/\lambda$）。多普勒是它的时间导数，所以常数项必须保留；`fmod 2π` 只是为了数值精度。
