# kalman-filter-lab

从零手写卡尔曼滤波，面向具身智能感知。

线性 KF → 扩展卡尔曼滤波（EKF）→ 无迹卡尔曼滤波（UKF）→ 粒子滤波（PF）。每种滤波器独立可运行，配有可视化输出和面向小白的详细中文教程。

## 快速开始

```bash
# 安装依赖（仅需 NumPy + Matplotlib）
pip install -r requirements.txt

# 运行全部滤波器对比（推荐）
python demos/demo_all.py

# 或单独运行
python demos/demo_kf.py      # 线性卡尔曼滤波
python demos/demo_ekf.py     # 扩展卡尔曼滤波
python demos/demo_ukf.py     # 无迹卡尔曼滤波
python demos/demo_pf.py      # 粒子滤波
```

## 滤波器对比

| 滤波器 | 观测模型 | 核心技巧 | 适用场景 |
|--------|---------|---------|---------|
| **KF** | 笛卡尔坐标 `[x, y]` | 线性高斯最优解 | 传感器直接输出位置 |
| **EKF** | 雷达 `[距离, 方位角]` | 雅可比矩阵局部线性化 | 弱非线性系统 |
| **UKF** | 雷达 `[距离, 方位角]` | Sigma 点采样（无需求导） | 强非线性系统 |
| **PF** | 雷达 `[距离, 方位角]` | 加权粒子近似任意分布 | 多模态/非高斯分布 |

## 输出文件

所有图片输出到 `output/` 目录：

| 文件 | 说明 |
|------|------|
| `kf_static.png` | KF 轨迹对比 + 误差曲线 |
| `ekf_static.png` | EKF 轨迹对比 + 误差曲线 |
| `ukf_static.png` | UKF 轨迹对比 + 误差曲线 |
| `pf_static.png` | PF 轨迹对比 + 误差曲线 |
| `*_animated.gif` | 对应滤波器的追踪动画 |
| `all_filters_comparison.png` | 四合一对比图 + RMSE 柱状图 |

## 项目结构

```
kalman-filter-lab/
├── filters/                     # 滤波器核心实现
│   ├── kalman_filter.py         # 线性 KF（标准五公式）
│   ├── extended_kf.py           # EKF（雅可比 + 角度环绕处理）
│   ├── unscented_kf.py          # UKF（sigma 点 + 无迹变换）
│   └── particle_filter.py       # PF（系统重采样 + 对数域权重）
├── utils/
│   ├── simulation.py            # 真实轨迹 + 线性/雷达两类观测
│   └── visualization.py         # 深色主题绘图 + 动画 + 四合一对比
├── demos/
│   ├── demo_kf.py               # 线性 KF 独立演示
│   ├── demo_ekf.py              # EKF 独立演示
│   ├── demo_ukf.py              # UKF 独立演示
│   ├── demo_pf.py               # PF 独立演示
│   └── demo_all.py              # 四合一对比
├── output/                      # 生成的图片和动画
├── tutorial.md                  # 面向小白的中文教程
└── README.md
```

## 原理概览

所有滤波器共享同一个 **2D 匀速运动模型**，追踪同一条微弯轨迹：

1. **真实轨迹** — 匀速运动 + 微小弯曲，作为"上帝视角"的地面真值
2. **传感器观测** — KF 用笛卡尔坐标（模拟 GPS），EKF/UKF/PF 用雷达距离-方位角（模拟激光雷达）
3. **滤波** — 每个滤波器用自己的方式融合"模型预测"和"传感器观测"
4. **对比** — `demo_all.py` 将四者放在同一张图上横向对比

## 依赖

- Python 3.10+
- NumPy
- Matplotlib

## 教程

`tutorial.md` 面向零基础读者，涵盖：

- 直觉理解（用日常类比而非公式堆砌）
- 数学准备（高斯分布、协方差、矩阵乘法）
- 线性 KF 五公式逐条拆解
- **EKF**：雅可比矩阵 = "在曲线上画切线"
- **UKF**：Sigma 点 = "派侦察兵去探路"
- **PF**：粒子 = "用一群蚂蚁找位置"
- 四大家族横向对比 + 决策指南
- 参数调优实验 + 常见问题
