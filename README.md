# isaaclab-lee-controller

Lee position controller for quadrotor UAVs in **NVIDIA Isaac Lab** — geometric tracking control on SE(3) with cascaded PD position/attitude loops.

Designed as a drop-in flight controller for Vision-Language-Action (VLA) guided autonomous drone navigation. Give it a `(dx, dy, dz)` command from your VLA model and it handles the rest.

---

## Overview

The controller follows the three-layer cascaded architecture from the original paper:

```
VLA model output: dx, dy, dz, dyaw
          │
          ▼
┌─────────────────────┐
│  Layer 1            │  Position PD
│  e_pos, e_vel  ──▶  │  Desired thrust vector (world frame)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Layer 2            │  Attitude PD (Lee, SE(3))
│  e_R,   e_ω    ──▶  │  Desired torque (body frame)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Layer 3            │  Actuation
│  F, M          ──▶  │  set_external_force_and_torque
└─────────────────────┘
```

A real quadrotor can only push along its body +z axis. The controller converts the world-frame desired force into a scalar thrust `T = F_des · b3`, so horizontal motion arises naturally from the attitude loop tilting the body. This avoids the common pitfall of applying arbitrary world-frame forces.

**Reference:**
> T. Lee, M. Leok, and N. H. McClamroch, "Geometric Tracking Control of a Quadrotor UAV on SE(3)," *Proc. IEEE CDC*, 2010, pp. 5420–5425.

Implementation inspired by [OmniDrones](https://github.com/btx0424/OmniDrones) and [ETH RotorS](https://github.com/ethz-asl/rotors_simulator).

---

## Requirements

| Package | Tested version |
|---------|---------------|
| NVIDIA Isaac Lab | Isaac Sim 6.0.0 |
| PyTorch | 2.x |
| Python | 3.12 |

Drone asset: **Crazyflie cf2x.usd** (available via Isaac Lab assets or the [Crazyflie firmware repo](https://github.com/bitcraze/crazyflie-firmware)).

---

## Quick Start

```python
from DroneController import DroneController

# 1. Create controller (default gains tuned for Crazyflie)
controller = DroneController(device="cuda:0")

# 2. Call reset() after every env.reset()
controller.reset(drone)  # drone: Isaac Lab Articulation

# 3. Update target from VLA output (world-frame delta)
controller.set_target_delta(dx=2.0, dy=0.0, dz=0.5)

# 4. Call step() every physics tick, before sim.step()
for step in range(num_steps):
    dist = controller.step(drone, dt=0.005)
    drone.write_data_to_sim()
    sim.step()
    drone.update(dt)
```

---

## API

### `DroneController(device, **gains)`

**Physical parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gravity` | `9.81` | Gravitational acceleration (m/s²) |
| `arm_length` | `0.046` | Rotor arm length (m); reserved for future motor allocation |
| `km_kf_ratio` | `0.006` | Torque-to-thrust coefficient ratio; reserved for future motor allocation |
| `Ixx` | `1.4e-5` | Moment of inertia about body x-axis (kg·m²) |
| `Iyy` | `1.4e-5` | Moment of inertia about body y-axis (kg·m²) |
| `Izz` | `2.17e-5` | Moment of inertia about body z-axis (kg·m²) |

**Position PD gains**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Kp_xy` | `1.2` | Position P gain, horizontal |
| `Kd_xy` | `1.8` | Velocity D gain, horizontal |
| `Kp_z` | `2.0` | Position P gain, vertical |
| `Kd_z` | `2.5` | Velocity D gain, vertical |

**Attitude PD gains**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `KR_xy` | `8e-4` | Attitude error gain, roll/pitch (N·m) |
| `KR_z` | `4e-4` | Attitude error gain, yaw (N·m) |
| `Kw_xy` | `2e-4` | Angular velocity gain, roll/pitch |
| `Kw_z` | `1e-4` | Angular velocity gain, yaw |

**Safety limits**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tilt_angle` | `0.35` rad | ~20°, limits horizontal aggressiveness |
| `max_thrust_ratio` | `3.0` | Max total thrust = 3 × hover thrust |
| `min_thrust_ratio` | `0.1` | Minimum vertical force as a fraction of `mass * gravity`; prevents attitude flip on overshoot |

**Logging**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `verbose` | `False` | Print target updates and reset banners to stdout |

### Methods

```python
controller.reset(drone)
# Initialize from current drone pose. Reads true mass from PhysX.
# Must be called after every env.reset().

controller.set_target_delta(dx, dy, dz, dyaw=0.0)
# Update target by world-frame delta. Targets are chained (cumulative).
# dyaw in radians; positive = counter-clockwise viewed from above.

controller.set_target_absolute(x, y, z, yaw=None)
# Set an absolute world-frame target position.
# yaw (rad): if None, current target yaw is preserved.

dist = controller.step(drone, dt=0.005)
# Run one control step. Returns mean distance to target (m).
# Call before sim.step() every physics tick.

err = controller.yaw_error(drone)
# Returns shortest yaw error per environment, shape (N,), values in [-pi, pi].
```

### Properties

```python
controller.target          # Current target position, shape (1, 3), world frame
controller.target_yaw      # Current target yaw, shape (1,), radians (wrapped)
controller.target_yaw_deg  # Current target yaw in degrees (float)
controller.drone_mass      # Drone mass in kg as read from PhysX at reset (float)
controller.last_thrust     # Total thrust applied at the previous step, shape (N,) in Newtons
                           # Returns None before the first step.
controller.last_F_total    # Alias of last_thrust (backward compatibility)
```

---

## Running the Test

```bash
# Headless test (no video)
python test_drone_controller.py --headless --device cuda:0

# With video recording
python test_drone_controller.py --headless --enable_cameras --device cuda:0 --video

# Custom USD path
python test_drone_controller.py --headless --device cuda:0 \
    --usd_path /path/to/cf2x.usd
```

Additional CLI options:

| Flag | Default | Description |
|------|---------|-------------|
| `--arrive_threshold` | `0.15` m | Position arrival threshold |
| `--yaw_threshold` | `5.0` deg | Yaw arrival threshold |
| `--hover_secs` | `2.0` s | Hover duration after each waypoint |
| `--timeout_secs` | `10.0` s | Per-waypoint timeout |
| `--video_dir` | `./output` | Directory for the saved video |

The test runs 7 waypoints covering translation, ascent, yaw rotation, and return to origin. Passing criteria: position error < 0.15 m and yaw error < 5° at each waypoint.

---

## Integration with VLA Models

The interface mirrors the output format of UAV navigation datasets such as **UAV-Flow**, where the model predicts `(dx, dy, dz, dyaw)` increments in the world frame:

```python
# High-level VLA + controller loop
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

## Citation

If you use this controller in academic work, please cite the original paper:

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

## License

BSD-3-Clause
