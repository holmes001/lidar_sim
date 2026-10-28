#ifndef LIDAR_TRANSPORT_H
#define LIDAR_TRANSPORT_H

#include <vector>
#include <complex>

constexpr double kpi = 3.14159265358979323846;
const double c = 3.0e8;

/*
 * 单位约定：全项目统一 SI。
 *   距离/高度/波长 m；面积 m^2；时间 s；频率 Hz；功率 W；幅度 sqrt(W)（与 tx_p 开方一致）。
 * 由此推出的两个指标：
 *   距离分辨率     = c*dt/2
 *   最大不模糊距离 = samp*dt*c/2
 */
typedef struct
{
    int samp;
    double tx_p,tx_area,rx_area,wavelenth,dt,height;

    /* 接收机噪声：每个正交分量（I/Q）的标准差，单位与信号幅度相同。
     * 由接收机决定，与目标距离、目标是否存在都无关 —— 这是 R^-4 定律成立的前提。 */
    double noise_sigma;
}radar;

typedef struct
{
    double area,distance,theta,velocity,height;
}object;

typedef enum { WF_PULSE, WF_SINE, WF_SQUARE, WF_IMPULSE } waveform_kind;

typedef struct {
    waveform_kind kind;

    union {
        struct { double width_s; double t_start_s; } pulse;    // 矩形脉冲
        struct { double period_s; }                  sine;     // 单频
        struct { double period_s; double duty; }     square;   // 方波

        /* 冲激串（Dirac 梳）：u(t) = peak_amp * Σ δ(t - t_start - k*period)，k = 0,1,2,...
         * 采样后表现为每个周期最多一个非零样点（幅度 = peak_amp），其余为 0。
         * period_s 应不小于采样间隔 dt，否则一个采样点会跨过多个冲激（混叠）。 */
        struct { double period_s; double t_start_s; } impulse;
    } param;

    double peak_amp;   // 峰值幅度（功率由图1.1 的规则从数据测得）
} waveform;


/**
 * @brief 雷达方程里的衰减因子 A = At*Ar*sigma / (4*pi*lambda^2*R^4)
 * @note  纯公式，只依赖物理量，因此不接收结构体
 */
double decay_factor(double tx_area, double rx_area, double obj_area,
                    double wavelength, double distance);

/** @brief 单程时延 R/c [s] */
double delay(double R);

/** @brief 往返时延 2R/c [s] */
double round_trip_delay(double R);

/* 波形生成需要采样上下文（samp / dt / tx_p），所以传 const 引用。
 * 按 waveform.kind 生成：脉冲/单频/冲激串均按连续时间表达式在 t = i*dt 处求值。
 * 冲激串只在冲激时刻与采样时刻重合（相差不超过 dt/2）的样点上取峰值。 */
void generate_a_signal(std::vector<std::complex<double>> &u,const radar &radar_1,waveform wave_1);

/**
 * @brief 回波信号：按连续时间求值时延、做 sqrt(decay_factor) 幅度衰减、叠加接收机噪声
 * @param u        发射波形（保留以兼容旧调用；回波改为按 wave 解析求值，不再从 u 搬移）
 * @param radar_1  雷达参数（含 noise_sigma）
 * @param object_1 目标参数
 * @param wave     波形参数（提供任意时刻 t 的幅度）
 * @return 长度为 radar_1.samp 的回波
 * @note  时延是连续量，不四舍五入到整数采样点，所以测距精度不受采样量化限制。
 */
std::vector<std::complex<double>> recieve_signal(
    const std::vector<std::complex<double>> &u,const radar &radar_1,const object &object_1,
    const waveform &wave);

#endif
