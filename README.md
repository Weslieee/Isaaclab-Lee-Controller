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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Kp_xy` | `2.0` | Position P gain, horizontal |
| `Kd_xy` | `2.5` | Velocity D gain, horizontal |
| `Kp_z` | `3.0` | Position P gain, vertical |
| `Kd_z` | `3.0` | Velocity D gain, vertical |
| `KR_xy` | `8e-4` | Attitude error gain, roll/pitch (N·m) |
| `KR_z` | `4e-4` | Attitude error gain, yaw (N·m) |
| `Kw_xy` | `2e-4` | Angular velocity gain, roll/pitch |
| `Kw_z` | `1e-4` | Angular velocity gain, yaw |
| `max_tilt_angle` | `0.5` rad | ~28°, limits horizontal aggressiveness |
| `max_thrust_ratio` | `3.0` | Max total thrust = 3 × hover thrust |

### Methods

```python
controller.reset(drone)
# Initialize from current drone pose. Reads true mass from PhysX.
# Must be called after every env.reset().

controller.set_target_delta(dx, dy, dz, dyaw=0.0)
# Update target by world-frame delta. Targets are chained (cumulative).
# dyaw in radians; positive = counter-clockwise viewed from above.

controller.set_target_absolute(x, y, z)
# Set an absolute world-frame target position.

dist = controller.step(drone, dt=0.005)
# Run one control step. Returns distance to target (m).
# Call before sim.step() every physics tick.
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
