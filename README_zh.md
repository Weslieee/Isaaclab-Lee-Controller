# isaaclab-lee-controller

基于 **NVIDIA Isaac Lab** 的四旋翼无人机 Lee 位置控制器 —— 在 SE(3) 上实现几何跟踪控制，采用级联 PD 位置/姿态双环结构。

专为视觉-语言-动作（VLA）引导的自主无人机导航设计，作为即插即用的飞行控制器。只需将 VLA 模型输出的 `(dx, dy, dz)` 指令传入，控制器负责其余所有计算。

---

## 概述

控制器遵循原论文的三层级联架构：

```
VLA 模型输出：dx, dy, dz, dyaw
          │
          ▼
┌─────────────────────┐
│  第一层             │  位置 PD
│  e_pos, e_vel  ──▶  │  期望推力向量（世界系）
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  第二层             │  姿态 PD（Lee，SE(3)）
│  e_R,   e_ω    ──▶  │  期望力矩（机体系）
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  第三层             │  执行器
│  F, M          ──▶  │  set_external_force_and_torque
└─────────────────────┘
```

真实四旋翼只能沿机体 +z 轴产生推力。控制器将世界系期望力投影为标量推力 `T = F_des · b3`，水平运动通过姿态环倾斜机体自然产生，从而避免了直接施加任意世界系力导致飞行器无倾斜平移的常见问题。

**参考文献：**
> T. Lee, M. Leok, and N. H. McClamroch, "Geometric Tracking Control of a Quadrotor UAV on SE(3)," *Proc. IEEE CDC*, 2010, pp. 5420–5425.

实现参考了 [OmniDrones](https://github.com/btx0424/OmniDrones) 和 [ETH RotorS](https://github.com/ethz-asl/rotors_simulator)。

---

## 依赖环境

| 软件包 | 测试版本 |
|--------|---------|
| NVIDIA Isaac Lab | Isaac Sim 6.0.0 |
| PyTorch | 2.x |
| Python | 3.12 |

无人机资产：**Crazyflie cf2x.usd**（可通过 Isaac Lab 资产库或 [Crazyflie 固件仓库](https://github.com/bitcraze/crazyflie-firmware) 获取）。

---

## 快速上手

```python
from DroneController import DroneController

# 1. 创建控制器（默认增益针对 Crazyflie 调参）
controller = DroneController(device="cuda:0")

# 2. 每次 env.reset() 后调用 reset()
controller.reset(drone)  # drone: Isaac Lab Articulation 对象

# 3. 传入 VLA 模型输出的世界系增量目标
controller.set_target_delta(dx=2.0, dy=0.0, dz=0.5)

# 4. 每个物理步调用 step()，需在 sim.step() 之前
for step in range(num_steps):
    dist = controller.step(drone, dt=0.005)
    drone.write_data_to_sim()
    sim.step()
    drone.update(dt)
```

---

## API

### `DroneController(device, **gains)`

**物理参数**

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `gravity` | `9.81` | 重力加速度（m/s²） |
| `arm_length` | `0.046` | 旋翼臂长（m）；预留，用于未来电机分配 |
| `km_kf_ratio` | `0.006` | 力矩推力系数比；预留，用于未来电机分配 |
| `Ixx` | `1.4e-5` | 机体 x 轴转动惯量（kg·m²） |
| `Iyy` | `1.4e-5` | 机体 y 轴转动惯量（kg·m²） |
| `Izz` | `2.17e-5` | 机体 z 轴转动惯量（kg·m²） |

**位置 PD 增益**

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `Kp_xy` | `1.2` | 水平位置比例增益 |
| `Kd_xy` | `1.8` | 水平速度微分增益 |
| `Kp_z` | `2.0` | 垂直位置比例增益 |
| `Kd_z` | `2.5` | 垂直速度微分增益 |

**姿态 PD 增益**

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `KR_xy` | `8e-4` | 横滚/俯仰姿态误差增益（N·m） |
| `KR_z` | `4e-4` | 偏航姿态误差增益（N·m） |
| `Kw_xy` | `2e-4` | 横滚/俯仰角速度增益 |
| `Kw_z` | `1e-4` | 偏航角速度增益 |

**安全限制**

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `max_tilt_angle` | `0.35` rad | 约 20°，限制水平机动幅度 |
| `max_thrust_ratio` | `3.0` | 最大总推力 = 3 × 悬停推力 |
| `min_thrust_ratio` | `0.1` | 期望力垂直分量的最小值（占 `mass * gravity` 的比例）；防止目标点越冲时姿态翻转 |

**日志**

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `verbose` | `False` | 是否向标准输出打印目标更新与 reset 信息 |

### 方法

```python
controller.reset(drone)
# 从当前无人机位姿初始化控制器，从 PhysX 读取真实质量。
# 每次 env.reset() 后必须调用。

controller.set_target_delta(dx, dy, dz, dyaw=0.0)
# 以世界系增量更新目标，每次调用在上一个目标基础上累加。
# dyaw 单位为弧度，正值 = 从上方俯视逆时针。

controller.set_target_absolute(x, y, z, yaw=None)
# 设置世界系绝对目标位置。
# yaw（rad）：若为 None，则保持当前目标偏航不变。

dist = controller.step(drone, dt=0.005)
# 执行一步控制，返回无人机到目标的平均距离（m）。
# 须在每个物理步的 sim.step() 之前调用。

err = controller.yaw_error(drone)
# 返回每个环境的最短偏航误差，shape (N,)，值域 [-pi, pi]。
```

### 属性

```python
controller.target          # 当前目标位置，shape (1, 3)，世界系
controller.target_yaw      # 当前目标偏航，shape (1,)，弧度（已归一化）
controller.target_yaw_deg  # 当前目标偏航，角度（float）
controller.drone_mass      # reset 时从 PhysX 读取的无人机质量，单位 kg（float）
controller.last_thrust     # 上一步施加的总推力，shape (N,)，单位 N
                           # 首次 step() 前为 None
controller.last_F_total    # last_thrust 的向后兼容别名
```

---

## 运行测试

```bash
# 无头模式测试（无视频）
python test_drone_controller.py --headless --device cuda:0

# 同时录制视频
python test_drone_controller.py --headless --enable_cameras --device cuda:0 --video

# 指定自定义 USD 路径
python test_drone_controller.py --headless --device cuda:0 \
    --usd_path /path/to/cf2x.usd
```

其他 CLI 参数：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--arrive_threshold` | `0.15` m | 位置到达判定阈值 |
| `--yaw_threshold` | `5.0` deg | 偏航到达判定阈值 |
| `--hover_secs` | `2.0` s | 每个航点到达后的悬停时长 |
| `--timeout_secs` | `10.0` s | 每个航点的超时时间 |
| `--video_dir` | `./output` | 视频保存目录 |

测试包含 7 个航点，覆盖平移、上升、偏航旋转及返回原点。通过标准：每个航点处位置误差 < 0.15 m，偏航误差 < 5°。

---

## 与 VLA 模型集成

接口格式与 **UAV-Flow** 等无人机导航数据集保持一致，模型预测世界系增量 `(dx, dy, dz, dyaw)`：

```python
# VLA + 控制器高层循环
for step in range(total_steps):
    if step % vla_interval == 0:
        image = get_camera_image()
        action = vla_model.predict(image, instruction)
        controller.set_target_delta(*action)   # dx, dy, dz, dyaw

    dist = controller.step(drone)
    drone.write_data_to_sim()
    sim.step()
    drone.update(dt)
```

---

## 引用

如在学术工作中使用本控制器，请引用原论文：

```bibtex
@inproceedings{lee2010geometric,
  title     = {Geometric Tracking Control of a Quadrotor {UAV} on {SE(3)}},
  author    = {Lee, Taeyoung and Leok, Melvin and McClamroch, N. Harris},
  booktitle = {Proc. IEEE Conference on Decision and Control (CDC)},
  pages     = {5420--5425},
  year      = {2010}
}
```

---

## 许可证

BSD-3-Clause
