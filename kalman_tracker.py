#!/usr/bin/env python3
"""二维卡尔曼滤波演示 —— 匀速目标追踪。

本脚本生成一条模拟的真实运动轨迹，用高斯噪声污染后模拟传感器读数，
然后用线性卡尔曼滤波器从噪声观测中恢复真实状态。
输出一张静态对比图和一段追踪动画 GIF。

用法：
    conda activate /home/amanannn/miniconda_envs/kf_demo
    python kalman_tracker.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头模式：只写文件，不弹出 GUI 窗口
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════════════════
# 仿真参数
# ═══════════════════════════════════════════════════════════════════════════════
DT = 0.1            # 时间步长 [秒]
TOTAL_TIME = 12.0   # 总仿真时间 [秒]
STEPS = int(TOTAL_TIME / DT)  # 总步数

# 真实轨迹：匀速运动 + 轻微加速度偏置，形成一条微弯的曲线
# 物体从原点 (0,0) 出发，主要向右移动，同时缓慢向下弯曲
TRUE_VX, TRUE_VY = 0.8, 0.25        # 基础速度 [米/秒]
CURVE_AX, CURVE_AY = 0.0, -0.02     # 微小向下加速度 [米/秒²]

# 传感器测量噪声标准差（x 和 y 方向相同）
SENSOR_NOISE = 0.45  # 米 —— 故意设得较大，以凸显滤波效果

# 卡尔曼滤波器过程噪声（我们对"模型不完美"的容忍程度）
Q_POS = 0.01   # 位置维度的过程噪声方差
Q_VEL = 0.05   # 速度维度的过程噪声方差


# ═══════════════════════════════════════════════════════════════════════════════
# 第 1 部分：生成真实轨迹（地面真值）
# ═══════════════════════════════════════════════════════════════════════════════
def generate_ground_truth():
    """生成真实的位置序列和速度序列。

    返回:
        pos: (STEPS, 2) 形状的数组，每行是 [x, y] 位置
        vel: (STEPS, 2) 形状的数组，每行是 [vx, vy] 速度
    """
    pos = np.zeros((STEPS, 2))
    vel = np.zeros((STEPS, 2))

    for k in range(1, STEPS):
        t = k * DT
        # 速度随时间缓慢变化（恒定加速度模型）
        vx = TRUE_VX + CURVE_AX * t
        vy = TRUE_VY + CURVE_AY * t
        vel[k] = [vx, vy]
        # 位置更新：p_new = p_old + v * dt
        pos[k] = pos[k - 1] + vel[k] * DT

    return pos, vel


# ═══════════════════════════════════════════════════════════════════════════════
# 第 2 部分：模拟噪声传感器观测
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_observations(true_pos):
    """在真实位置上叠加高斯噪声，模拟不完美的传感器读数。

    真实传感器（GPS、摄像头位姿估计等）永远有误差。
    这里用零均值高斯噪声来近似传感器的随机误差。

    参数:
        true_pos: 真实位置数组
    返回:
        带噪声的观测位置数组（模拟传感器读数）
    """
    rng = np.random.default_rng(42)  # 固定种子，保证结果可复现
    noise = rng.normal(0, SENSOR_NOISE, size=true_pos.shape)
    return true_pos + noise


# ═══════════════════════════════════════════════════════════════════════════════
# 第 3 部分：卡尔曼滤波器核心实现
# ═══════════════════════════════════════════════════════════════════════════════
class KalmanFilter:
    """线性卡尔曼滤波器 —— 2D 匀速运动模型。

    状态向量 x: [位置_x, 位置_y, 速度_x, 速度_y]ᵀ   (4×1)
    观测向量 z:  [观测位置_x, 观测位置_y]ᵀ           (2×1)

    滤波器每收到一个新的传感器读数，执行一次"预测→更新"循环。
    """

    def __init__(self, dt: float):
        """初始化卡尔曼滤波器的所有矩阵。

        参数:
            dt: 时间步长 [秒]
        """
        self.dt = dt

        # ── 状态转移矩阵 F ──
        # 描述状态如何随时间演化：x_{k+1} = F · x_k
        # 匀速模型下，位置 = 旧位置 + 速度 × dt，速度不变
        #     px_new = px_old + vx_old * dt
        #     py_new = py_old + vy_old * dt
        #     vx_new = vx_old          (速度不变)
        #     vy_new = vy_old          (速度不变)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])

        # ── 观测矩阵 H ──
        # 描述状态如何映射到传感器读数：z = H · x
        # 我们的传感器只能观测位置，不能直接测速度
        #     z_x = px  (观测到 x 位置)
        #     z_y = py  (观测到 y 位置)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # ── 过程噪声协方差矩阵 Q ──
        # 量化"运动模型有多不完美"
        # Q 越大 = 我们越不相信模型的预测，滤波器会更多依赖传感器
        # Q 越小 = 我们越相信模型，滤波器会更平滑但可能跟不上突变
        q_diag = np.array([Q_POS, Q_POS, Q_VEL, Q_VEL])
        self.Q = np.diag(q_diag)

        # ── 测量噪声协方差矩阵 R ──
        # 量化"传感器有多不准"
        # 这里假设 x 和 y 方向的噪声独立且等强度
        self.R = np.eye(2) * (SENSOR_NOISE ** 2)

        # ── 初始状态估计 ──
        # 我们不知道物体在哪儿，猜它在原点，速度为零
        self.x = np.zeros((4, 1))
        # 初始协方差矩阵设得偏大，表示"我们对初始猜测很不确定"
        # 这样滤波器的前几步会大幅修正，快速靠近真实值
        self.P = np.eye(4) * 0.5

        # 存储每一步的滤波结果，用于后续可视化
        self.estimates = []     # 每步估计的 (位置_x, 位置_y)
        self.covariances = []   # 每步完整的 4×4 协方差矩阵 P

    def predict(self):
        """预测步骤（卡尔曼滤波公式 1 和 2）。

        公式 1：x = F · x            —— 用运动模型把状态推到下一时刻
        公式 2：P = F · P · Fᵀ + Q  —— 不确定性也跟着传播，而且因为模型
                                        不完美，还要额外加上过程噪声 Q
        """
        self.x = self.F @ self.x               # 公式 1：状态预测
        self.P = self.F @ self.P @ self.F.T + self.Q  # 公式 2：协方差预测

    def update(self, z: np.ndarray):
        """更新步骤（卡尔曼滤波公式 3、4、5）。

        传感器读数到了，我们需要用它来修正预测结果。

        参数:
            z: 传感器读数 [观测_x, 观测_y]，形状 (2,)
        """
        # 创新（残差）：传感器实际读数 减去 我们预测的读数
        # 如果预测完美，残差应该接近于零
        y = z.reshape(2, 1) - self.H @ self.x

        # 创新协方差：残差的不确定程度
        S = self.H @ self.P @ self.H.T + self.R

        # 公式 3：卡尔曼增益 K —— 核心权重！
        # K 决定了"传感器读数"和"模型预测"之间的信任分配
        # 如果传感器噪声大(R大)，K 自动变小 → 少信传感器
        # 如果模型不确定(P大)，K 自动变大 → 多信传感器
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 公式 4：用观测修正状态估计
        self.x = self.x + K @ y

        # 公式 5：修正后，不确定性降低了
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def step(self, z: np.ndarray):
        """执行一次完整的"预测→更新"循环。"""
        self.predict()
        self.update(z)
        # 记录当前估计结果
        self.estimates.append(self.x[:2, 0].copy())
        self.covariances.append(self.P.copy())


def run_filter(observations):
    """对所有观测数据依次运行卡尔曼滤波器。

    参数:
        observations: (STEPS, 2) 形状的观测数组
    返回:
        estimates:   (STEPS, 2) 形状的估计位置数组
        covariances: 每一步的 4×4 协方差矩阵列表
    """
    kf = KalmanFilter(DT)
    for z in observations:
        kf.step(z)
    return np.array(kf.estimates), kf.covariances


# ═══════════════════════════════════════════════════════════════════════════════
# 第 4 部分：静态可视化
# ═══════════════════════════════════════════════════════════════════════════════
def uncertainty_ellipse(cov_2x2, n_std=1.0):
    """从 2×2 协方差矩阵提取不确定性椭圆的参数。

    协方差矩阵的特征值代表椭圆各轴的长度，
    特征向量代表椭圆各轴的方向。

    参数:
        cov_2x2: 位置的 2×2 协方差子矩阵
        n_std:   几个标准差（1σ = 68% 置信区间）
    返回:
        (宽度, 高度, 角度_度)
    """
    eigvals, eigvecs = np.linalg.eigh(cov_2x2)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    return width, height, angle


def draw_static_figure(true_pos, obs, est, covs):
    """绘制并保存静态对比图（深色主题）。

    左子图：三条轨迹线 + 不确定性椭圆
    右子图：观测误差和滤波误差随时间变化的对比曲线
    """
    plt.style.use("dark_background")

    colors = {"true": "#00ff88", "obs": "#ff4444", "est": "#4488ff",
              "ellipse": "#4488ff", "grid": "#222222"}

    fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(16, 7),
                                           facecolor="#0a0a0a")
    fig.suptitle("Kalman Filter — 2D Constant Velocity Tracking",
                 fontsize=18, fontweight="bold", color="#cccccc", y=0.97)

    # ── 左子图：轨迹对比 ──
    ax_traj.set_facecolor("#0a0a0a")
    ax_traj.plot(true_pos[:, 0], true_pos[:, 1],
                 color=colors["true"], linewidth=2, label="Ground Truth")
    ax_traj.plot(obs[:, 0], obs[:, 1],
                 color=colors["obs"], linewidth=0.7, alpha=0.6,
                 label="Noisy Observation")
    ax_traj.plot(est[:, 0], est[:, 1],
                 color=colors["est"], linewidth=2, label="Kalman Estimate")

    # 每隔一段距离画一个不确定性椭圆（1σ 椭圆）
    step_skip = max(1, STEPS // 18)
    for i in range(0, STEPS, step_skip):
        cov_2x2 = covs[i][:2, :2]  # 从 4×4 P 中取出位置的 2×2 子矩阵
        w, h, ang = uncertainty_ellipse(cov_2x2, n_std=1.0)
        if w < 1e-6:
            continue
        e = Ellipse((est[i, 0], est[i, 1]), w, h, angle=ang,
                    facecolor=colors["ellipse"], edgecolor=colors["ellipse"],
                    alpha=0.08, linewidth=0.5)
        ax_traj.add_patch(e)

    ax_traj.set_xlabel("X position [m]", color="#aaaaaa")
    ax_traj.set_ylabel("Y position [m]", color="#aaaaaa")
    ax_traj.legend(loc="upper right", facecolor="#111111",
                   edgecolor="#333333", labelcolor="#cccccc")
    ax_traj.set_title("Trajectory Comparison", color="#bbbbbb", fontsize=13)
    ax_traj.grid(True, color=colors["grid"], alpha=0.4)
    ax_traj.set_aspect("equal")

    # ── 右子图：误差随时间变化 ──
    time = np.arange(STEPS) * DT

    obs_error = np.linalg.norm(obs - true_pos, axis=1)  # 观测误差（欧几里得距离）
    kf_error = np.linalg.norm(est - true_pos, axis=1)   # 滤波误差（欧几里得距离）

    ax_err.set_facecolor("#0a0a0a")
    ax_err.fill_between(time, 0, obs_error, color=colors["obs"],
                        alpha=0.12, label="Observation Error Area")
    ax_err.plot(time, obs_error, color=colors["obs"], linewidth=0.8,
                alpha=0.5, label="Observation Error")
    ax_err.plot(time, kf_error, color=colors["est"], linewidth=2,
                label="Kalman Error")

    # 标注 RMSE 和误差降低百分比
    obs_rmse = np.sqrt(np.mean(obs_error ** 2))
    kf_rmse = np.sqrt(np.mean(kf_error ** 2))
    reduction = (1 - kf_rmse / obs_rmse) * 100
    ax_err.text(0.98, 0.92,
                f"Obs RMSE: {obs_rmse:.3f} m\nKF  RMSE: {kf_rmse:.3f} m\n"
                f"Reduction: {reduction:.1f}%",
                transform=ax_err.transAxes, ha="right", va="top",
                fontsize=11, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#111111",
                          edgecolor="#444444", alpha=0.9),
                color="#cccccc")

    ax_err.set_xlabel("Time [s]", color="#aaaaaa")
    ax_err.set_ylabel("Position Error [m]", color="#aaaaaa")
    ax_err.set_title("Error Comparison Over Time", color="#bbbbbb", fontsize=13)
    ax_err.legend(loc="upper right", facecolor="#111111",
                  edgecolor="#333333", labelcolor="#cccccc")
    ax_err.grid(True, color=colors["grid"], alpha=0.4)

    out_path = OUT_DIR / "kalman_static.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] 静态对比图已保存 → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 第 5 部分：动画
# ═══════════════════════════════════════════════════════════════════════════════
def draw_animated_gif(true_pos, obs, est, covs):
    """绘制并保存追踪动画 GIF。

    动画展示了滤波器在每一步如何根据传感器读数更新估计，
    不确定性椭圆逐渐缩小，直观体现"信心增长"过程。
    """
    colors = {"true": "#00ff88", "obs": "#ff4444", "est": "#4488ff",
              "ellipse": "#4488ff", "grid": "#222222"}

    fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(16, 7),
                                           facecolor="#0a0a0a")
    fig.suptitle("Kalman Filter — Live Tracking", fontsize=18,
                 fontweight="bold", color="#cccccc", y=0.97)

    time = np.arange(STEPS) * DT

    # 创建空的艺术对象，供动画逐帧更新
    (line_true,) = ax_traj.plot([], [], color=colors["true"], linewidth=2,
                                label="Ground Truth")
    (line_obs,) = ax_traj.plot([], [], color=colors["obs"], linewidth=0.6,
                               alpha=0.4, label="Noisy Observation")
    (line_est,) = ax_traj.plot([], [], color=colors["est"], linewidth=2.5,
                               label="Kalman Estimate")
    (scat_true,) = ax_traj.plot([], [], "o", color=colors["true"],
                                markersize=8, markeredgecolor="white",
                                markeredgewidth=0.5)
    (scat_est,) = ax_traj.plot([], [], "o", color=colors["est"],
                               markersize=10, markeredgecolor="white",
                               markeredgewidth=0.5)
    ellipse_patch = Ellipse((0, 0), 0, 0, angle=0,
                            facecolor=colors["ellipse"], edgecolor=colors["ellipse"],
                            alpha=0.15, linewidth=0.5)
    ax_traj.add_patch(ellipse_patch)

    ax_traj.set_facecolor("#0a0a0a")
    ax_traj.set_xlabel("X position [m]", color="#aaaaaa")
    ax_traj.set_ylabel("Y position [m]", color="#aaaaaa")
    ax_traj.legend(loc="upper right", facecolor="#111111",
                   edgecolor="#333333", labelcolor="#cccccc")
    ax_traj.set_title("Trajectory", color="#bbbbbb", fontsize=13)
    ax_traj.grid(True, color=colors["grid"], alpha=0.4)
    ax_traj.set_aspect("equal")

    # 误差子图的初始设置
    obs_err_all = np.linalg.norm(obs - true_pos, axis=1)
    kf_err_all = np.linalg.norm(est - true_pos, axis=1)
    max_err = np.max(obs_err_all) * 1.05

    ax_err.set_facecolor("#0a0a0a")
    (line_obs_err,) = ax_err.plot([], [], color=colors["obs"], linewidth=0.6,
                                  alpha=0.4, label="Observation Error")
    (line_kf_err,) = ax_err.plot([], [], color=colors["est"], linewidth=2.5,
                                 label="Kalman Error")
    (scat_obs_err,) = ax_err.plot([], [], "o", color=colors["obs"],
                                  markersize=4, alpha=0.5)
    (scat_kf_err,) = ax_err.plot([], [], "o", color=colors["est"],
                                 markersize=5)
    ax_err.set_xlim(0, TOTAL_TIME)
    ax_err.set_ylim(0, max_err)
    ax_err.set_xlabel("Time [s]", color="#aaaaaa")
    ax_err.set_ylabel("Position Error [m]", color="#aaaaaa")
    ax_err.set_title("Error Over Time", color="#bbbbbb", fontsize=13)
    ax_err.legend(loc="upper right", facecolor="#111111",
                  edgecolor="#333333", labelcolor="#cccccc")
    ax_err.grid(True, color=colors["grid"], alpha=0.4)

    # 自动确定轨迹图的坐标范围
    all_x = np.concatenate([true_pos[:, 0], obs[:, 0], est[:, 0]])
    all_y = np.concatenate([true_pos[:, 1], obs[:, 1], est[:, 1]])
    pad_x = (all_x.max() - all_x.min()) * 0.1 + 0.5
    pad_y = (all_y.max() - all_y.min()) * 0.1 + 0.5
    ax_traj.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax_traj.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

    # 降采样帧数以控制 GIF 文件大小
    n_frames = min(STEPS, 150)
    stride = max(1, STEPS // n_frames)
    frames = list(range(0, STEPS, stride))
    if frames[-1] != STEPS - 1:
        frames.append(STEPS - 1)  # 确保最后一帧是最终状态

    def animate(i):
        """更新第 i 帧的所有艺术对象。"""
        idx = frames[i]
        traj_slice = slice(0, idx + 1)

        # 更新轨迹线（只显示从开始到当前时刻的部分）
        line_true.set_data(true_pos[traj_slice, 0], true_pos[traj_slice, 1])
        line_obs.set_data(obs[traj_slice, 0], obs[traj_slice, 1])
        line_est.set_data(est[traj_slice, 0], est[traj_slice, 1])
        scat_true.set_data([true_pos[idx, 0]], [true_pos[idx, 1]])
        scat_est.set_data([est[idx, 0]], [est[idx, 1]])

        # 更新不确定性椭圆
        cov_2x2 = covs[idx][:2, :2]
        w, h, ang = uncertainty_ellipse(cov_2x2, n_std=1.0)
        ellipse_patch.set_center((est[idx, 0], est[idx, 1]))
        ellipse_patch.width = max(w, 1e-6)
        ellipse_patch.height = max(h, 1e-6)
        ellipse_patch.angle = ang

        # 更新误差曲线
        t = time[traj_slice]
        line_obs_err.set_data(t, obs_err_all[traj_slice])
        line_kf_err.set_data(t, kf_err_all[traj_slice])
        scat_obs_err.set_data([time[idx]], [obs_err_all[idx]])
        scat_kf_err.set_data([time[idx]], [kf_err_all[idx]])

        return (line_true, line_obs, line_est, scat_true, scat_est,
                ellipse_patch, line_obs_err, line_kf_err,
                scat_obs_err, scat_kf_err)

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                   interval=80, blit=True)
    gif_path = OUT_DIR / "kalman_animated.gif"
    ani.save(gif_path, writer="pillow", fps=12, dpi=100)
    plt.close(fig)
    print(f"[✓] 追踪动画已保存 → {gif_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 第 6 部分：性能指标
# ═══════════════════════════════════════════════════════════════════════════════
def print_metrics(true_pos, obs, est):
    """在终端打印滤波效果汇总指标。

    RMSE（均方根误差）是最常用的误差衡量指标：
    值越小说明估计越接近真实值。
    """
    obs_err = np.linalg.norm(obs - true_pos, axis=1)
    kf_err = np.linalg.norm(est - true_pos, axis=1)

    obs_rmse = np.sqrt(np.mean(obs_err ** 2))
    kf_rmse = np.sqrt(np.mean(kf_err ** 2))
    reduction = (1 - kf_rmse / obs_rmse) * 100

    print("=" * 56)
    print("  二维卡尔曼滤波 — 匀速目标追踪演示")
    print("=" * 56)
    print(f"  仿真时长:        {TOTAL_TIME:.0f} 秒")
    print(f"  时间步长:        {DT:.2f} 秒  (共 {STEPS} 步)")
    print(f"  传感器噪声 (1σ): {SENSOR_NOISE:.2f} 米")
    print(f"  ──────────────────────────────────")
    print(f"  原始观测 RMSE:   {obs_rmse:.3f} 米")
    print(f"  卡尔曼滤波 RMSE: {kf_rmse:.3f} 米")
    print(f"  误差降低幅度:    {reduction:.1f}%")
    print("=" * 56)


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("→ 正在生成真实轨迹 …")
    true_pos, true_vel = generate_ground_truth()

    print("→ 正在模拟噪声传感器读数 …")
    observations = simulate_observations(true_pos)

    print("→ 正在运行卡尔曼滤波器 …")
    estimates, covariances = run_filter(observations)

    print_metrics(true_pos, observations, estimates)

    print("\n→ 正在渲染静态对比图 …")
    draw_static_figure(true_pos, observations, estimates, covariances)

    print("→ 正在渲染追踪动画（可能需要几秒钟）…")
    draw_animated_gif(true_pos, observations, estimates, covariances)

    print("\n✅ 全部完成！输出文件：")
    print(f"   📊 {OUT_DIR / 'kalman_static.png'}")
    print(f"   🎬 {OUT_DIR / 'kalman_animated.gif'}")


if __name__ == "__main__":
    main()
