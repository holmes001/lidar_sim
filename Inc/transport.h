/**
 * @file    transport.h
 * @brief   雷达链路：雷达方程、时延、波形生成与回波合成
 */
#ifndef LIDAR_TRANSPORT_H
#define LIDAR_TRANSPORT_H

#include <cstddef>

#include "complex_types.h"

constexpr double kpi = 3.14159265358979323846;   ///< 圆周率
const double c = 3.0e8;                          ///< 光速 [m/s]

/**
 * @brief 雷达参数
 *
 * @note 单位统一 SI：距离/高度/波长 m，面积 m^2，时间 s，频率 Hz，
 *       功率 W，幅度 sqrt(W)（与 tx_p 开方一致）。
 * @note 派生指标：距离分辨率 = c*dt/2；最大不模糊距离 = samp*dt*c/2。
 */
typedef struct
{
    std::size_t samp;                                ///< 采样点数
    double tx_p,tx_area,rx_area,wavelenth,dt,height; ///< 发射功率/收发面积/波长/采样间隔/高度
    double noise_sigma;                              ///< 每个 I/Q 分量的噪声标准差
}radar;

/**
 * @brief 目标参数
 *
 * @note velocity 为径向速度，符号决定多普勒方向（接近为正）。
 */
typedef struct
{
    double area,distance,theta,velocity,height;   ///< 截面积/距离/角度/径向速度/高度
}object;

/** @brief 波形种类 */
typedef enum { WF_PULSE, WF_SINE, WF_SQUARE, WF_IMPULSE, WF_FMCW } waveform_kind;

/**
 * @brief 波形参数，描述复包络 a(t)，而非收到的实信号 s(t)
 *
 * s(t) = Re{ a(t)·e^{j2πf₀t} }，f₀ = c/λ；|a| 为瞬时幅度，
 * arg(a) 为该时刻相对中心载波 f₀ 的相位。
 *
 * @warning param 是 union，只能读写当前 kind 对应的那个成员；
 *          连续给多个成员赋值会互相覆盖，跨成员读取属类型双关、结果无定义。
 * @warning 幅度调制（脉冲、CW）的包络为实数；只有瞬时频率偏离 f₀ 的调制
 *          （FMCW）包络才是复数。
 * @warning t_start_s 会整体平移匹配滤波峰位：峰位 = 回波延时 + t_start_s，
 *          测距标定时必须减掉。
 */
typedef struct {
    waveform_kind kind;   ///< 波形种类

    union {
        struct { double width_s; double t_start_s; } pulse;    ///< 矩形脉冲：a = peak_amp
        struct { double period_s; }                  sine;     ///< 单频连续波：a ≡ peak_amp（常数）
        struct { double period_s; double duty; }     square;   ///< 方波窗：a = ±peak_amp
        struct { double period_s; double t_start_s; } impulse; ///< 冲激串：每周期最多一个非零样点
        /**
         * 线性调频：a = peak_amp·exp(j·π·S·(tau² − tau·T))，tau ∈ [0, T)。
         * S = bandwidth_hz / sweep_s，瞬时基带频率 S·(tau − T/2) 从 −B/2 扫到 +B/2，
         * 因此 λ 指中心载波波长。
         */
        struct { double sweep_s; double bandwidth_hz; double t_start_s; } fmcw;
    } param;

    double peak_amp;   ///< 峰值幅度
} waveform;


/**
 * @brief 雷达方程衰减因子 A = At·Ar·σ/(4π·λ²·R⁴)
 * @param tx_area    发射面积 [m^2]
 * @param rx_area    接收面积 [m^2]
 * @param obj_area   目标截面积 [m^2]
 * @param wavelength 波长 [m]
 * @param distance   距离 [m]
 * @return 功率比，取 sqrt 后才是幅度因子
 * @note 纯公式，只依赖数值，便于单元测试与复用。
 */
double decay_factor(double tx_area, double rx_area, double obj_area,
                    double wavelength, double distance);

/** @brief 单程时延 R/c [s] */
double delay(double R);

/** @brief 往返时延 2R/c [s] */
double round_trip_delay(double R);

/**
 * @brief 生成发射复包络
 * @param[out] u       输出复包络，有效长度 min(u.size(), samp)
 * @param      radar_1 采样上下文（samp / dt）
 * @param      wave_1  波形参数
 */
void generate_a_signal(ComplexVec &u,const radar &radar_1,waveform wave_1);

/**
 * @brief 合成单目标回波
 * @param u        发射复包络（保留以兼容旧调用，当前不参与计算）
 * @param radar_1  雷达参数（含 noise_sigma）
 * @param object_1 目标参数
 * @param wave     波形参数
 * @param t_offset 记录起点的绝对时间偏移 [s]，第 p 个脉冲传 p·PRI
 * @return 长度为 samp 的回波复包络
 * @note 时延按连续时间求值，不量化到采样点；噪声只由 noise_sigma 决定，
 *       与距离及目标是否存在无关。
 * @warning t_offset 决定多普勒相位是否跨脉冲累进：恒传 0 会让每个脉冲的
 *          多普勒相位都从 0 重来，慢时间序列变成常量、多普勒恒为 0。
 */
ComplexVec recieve_signal(
    const ComplexVec &u,const radar &radar_1,const object &object_1,
    const waveform &wave,double t_offset = 0.0);

#endif
