# kalman-filter-lab

从零手写卡尔曼滤波，面向具身智能感知。

线性 KF 起步，逐步覆盖 EKF、UKF、粒子滤波。每个变体都配有代码、可视化和详细中文教程。

## 快速开始

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行 Demo
python kalman_tracker.py
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `kalman_static.png` | 静态对比图：轨迹对比 + 误差随时间变化曲线 |
| `kalman_animated.gif` | 追踪动画：滤波器实时收敛过程 |

终端同时输出 RMSE 指标，误差通常降低 50~60%。

## 项目结构

```
kalman-filter-lab/
├── kalman_tracker.py    # 线性卡尔曼滤波：仿真 + 滤波 + 可视化
├── tutorial.md          # 中文教程（直觉 → 数学 → 公式 → 代码 → 调参）
├── requirements.txt     # Python 依赖
└── README.md
```

## 原理概览

1. **真实轨迹** — 匀速运动 + 微弯曲线，作为"上帝视角"的地面真值
2. **噪声观测** — 叠加高斯噪声，模拟真实传感器读数
3. **卡尔曼滤波** — 预测（运动模型） → 更新（传感器修正）循环迭代
4. **可视化** — 轨迹对比 + 不确定性椭圆 + 误差曲线

## 调参实验

修改 `kalman_tracker.py` 顶部的常量即可实验：

| 参数 | 作用 |
|------|------|
| `SENSOR_NOISE` | 越大 → 观测越乱，滤波效果越明显 |
| `Q_POS / Q_VEL` | 越大 → 越信任传感器（响应更快，但更不平滑） |
| `CURVE_AX / CURVE_AY` | 轨迹弯曲度 → 测试滤波器在模型失配下的表现 |

## 依赖

- Python 3.10+
- NumPy
- Matplotlib

## 进阶路线

`tutorial.md` 中文教程涵盖：
- 卡尔曼滤波五个公式的直觉理解
- 数学基础（高斯分布、协方差矩阵、矩阵乘法）
- 代码逐行对照公式
- 参数调优实验指南
- 后续学习：EKF → UKF → 粒子滤波
