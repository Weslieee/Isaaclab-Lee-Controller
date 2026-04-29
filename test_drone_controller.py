"""Test script for DroneController in an empty Isaac Lab scene.

Runs a sequence of waypoints to validate position tracking and yaw control:

    1. Hover in place at (0, 0, 1.5)
    2. Move toward camera  (+0.8, -0.8,  0.0)
    3. Strafe left          ( 0.0, +0.8,  0.0)
    4. Ascend 0.5 m         ( 0.0,  0.0, +0.5)
    5. Yaw -60 deg (CW)     ( 0.0,  0.0,  0.0, -60 deg)
    6. Yaw +120 deg (CCW)   ( 0.0,  0.0,  0.0, +120 deg)
    7. Return to origin     (-0.8,  0.0, -0.5, -60 deg)

Usage (headless)::

    python test_drone_controller.py --headless --device cuda:0

Usage (with video recording)::

    python test_drone_controller.py --headless --enable_cameras --device cuda:0 --video

Requirements:
    - DroneController.py in the same directory
    - NVIDIA Isaac Lab with Isaac Sim 6.0.0
    - Crazyflie cf2x.usd asset at the path specified by --usd_path
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

parser = argparse.ArgumentParser(description="DroneController waypoint test.")
parser.add_argument(
    "--usd_path",
    type=str,
    default="/mnt/sdb/isaaclab/assets/Crazyflie/cf2x.usd",
    help="Path to the Crazyflie USD asset.",
)
parser.add_argument(
    "--arrive_threshold",
    type=float,
    default=0.15,
    help="Position arrival threshold (m). Default: 0.15",
)
parser.add_argument(
    "--yaw_threshold",
    type=float,
    default=5.0,
    help="Yaw arrival threshold (deg). Default: 5.0",
)
parser.add_argument(
    "--hover_secs",
    type=float,
    default=2.0,
    help="Hover duration after each waypoint (s). Default: 2.0",
)
parser.add_argument(
    "--timeout_secs",
    type=float,
    default=10.0,
    help="Per-waypoint timeout (s). Default: 10.0",
)
parser.add_argument(
    "--video",
    action="store_true",
    help="Record flight video (requires --enable_cameras).",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default="./output",
    help="Directory for saved video. Default: ./output",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------------ #
#  Imports (must come after AppLauncher)                              #
# ------------------------------------------------------------------ #

import torch
from PIL import Image as PILImage

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from DroneController import DroneController

# ------------------------------------------------------------------ #
#  Constants                                                          #
# ------------------------------------------------------------------ #

SIM_DT = 0.005   # Physics timestep (s) — 200 Hz
ROTOR_SPEED_K = 2200.0  # Visual spin scaling constant

# Waypoints: (description, dx, dy, dz, dyaw_deg)
# All deltas are relative to the previous target in world frame.
# dyaw_deg: positive = counter-clockwise viewed from above.
WAYPOINTS = [
    ("Hover in place",      0.0,  0.0,  0.0,    0.0),
    ("Move toward camera",  0.8, -0.8,  0.0,    0.0),
    ("Strafe left",         0.0,  0.8,  0.0,    0.0),
    ("Ascend 0.5 m",        0.0,  0.0,  0.5,    0.0),
    ("Yaw -60 deg (CW)",    0.0,  0.0,  0.0,  -60.0),
    ("Yaw +120 deg (CCW)",  0.0,  0.0,  0.0,  120.0),
    ("Return to origin",   -0.8,  0.0, -0.5,  -60.0),
]


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def log(msg: str) -> None:
    print(f"[TEST] {msg}", flush=True)


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main() -> None:
    # -- Simulation setup ------------------------------------------ #
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, -1.5, 1.8], target=[0.0, 0.0, 1.5])

    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0)
    )

    # -- Drone asset ----------------------------------------------- #
    drone_cfg = ArticulationCfg(
        prim_path="/World/Crazyflie",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args_cli.usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "rotors": ImplicitActuatorCfg(
                joint_names_expr=["m1_joint", "m2_joint", "m3_joint", "m4_joint"],
                effort_limit=1.0,
                velocity_limit=1000.0,
                stiffness=0.0,
                damping=1.0,
            ),
        },
    )
    drone_cfg.spawn.func(
        "/World/Crazyflie",
        drone_cfg.spawn,
        translation=drone_cfg.init_state.pos,
    )
    drone = Articulation(drone_cfg)

    # -- Video setup ----------------------------------------------- #
    rgb_annot = None
    video_frames = []
    video_interval = max(1, int(1.0 / (SIM_DT * 25)))  # target 25 fps

    if args_cli.video:
        import omni.replicator.core as rep
        os.makedirs(args_cli.video_dir, exist_ok=True)
        rp = rep.create.render_product("/OmniverseKit_Persp", (1920, 1080))
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb_annot.attach([rp])
        log(f"Video recording enabled -> {args_cli.video_dir}/drone_flight.mp4")

    # -- Start simulation ------------------------------------------ #
    sim.reset()
    drone.update(SIM_DT)

    # Rotor visual spin via USD Xform (bypasses physics)
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    prop_mesh_paths = [
        "/World/Crazyflie/m1_prop/ccw_prop",
        "/World/Crazyflie/m2_prop/cw_prop",
        "/World/Crazyflie/m3_prop/ccw_prop",
        "/World/Crazyflie/m4_prop/cw_prop",
    ]
    spin_ops = []
    for path in prop_mesh_paths:
        prim = stage.GetPrimAtPath(path)
        op = UsdGeom.Xformable(prim).AddRotateZOp(opSuffix="propSpin")
        spin_ops.append(op)

    rotor_angles_deg = [0.0, 0.0, 0.0, 0.0]
    rotor_dir = [1.0, -1.0, 1.0, -1.0]  # m1/m3 CCW, m2/m4 CW

    log("Simulation ready.")

    # -- Controller init ------------------------------------------- #
    controller = DroneController(device=args_cli.device)
    controller.reset(drone)

    # -- Waypoint loop --------------------------------------------- #
    arrive_th = args_cli.arrive_threshold
    arrive_yaw_th = math.radians(args_cli.yaw_threshold)
    hover_secs = args_cli.hover_secs
    timeout = args_cli.timeout_secs
    log_every = max(1, int(1.0 / SIM_DT))

    all_passed = True
    global_step = 0

    for wp_idx, (desc, dx, dy, dz, dyaw_deg) in enumerate(WAYPOINTS):
        dyaw = math.radians(dyaw_deg)
        log(f"\n{'=' * 60}")
        log(
            f"Waypoint {wp_idx + 1}/{len(WAYPOINTS)}: {desc}  "
            f"delta=({dx:.1f}, {dy:.1f}, {dz:.1f}, {dyaw_deg:.0f} deg)"
        )

        controller.set_target_delta(dx, dy, dz, dyaw)
        tgt = controller.target[0].tolist()
        tgt_yaw = controller.target_yaw_deg
        log(f"Target: pos=({tgt[0]:.2f}, {tgt[1]:.2f}, {tgt[2]:.2f}), yaw={tgt_yaw:.1f} deg")

        # -- Fly to waypoint --------------------------------------- #
        elapsed = 0.0
        arrived = False
        step_count = 0

        while elapsed < timeout:
            dist = controller.step(drone)
            drone.write_data_to_sim()
            sim.step()

            # Rotor visual spin
            if controller.last_F_total is not None:
                avg_thrust = controller.last_F_total[0].item() / 4.0
                spin_rate = math.degrees(
                    ROTOR_SPEED_K * math.sqrt(max(avg_thrust, 0.0))
                )
                for i in range(4):
                    rotor_angles_deg[i] += rotor_dir[i] * spin_rate * SIM_DT
                    spin_ops[i].Set(float(rotor_angles_deg[i]))

            drone.update(SIM_DT)

            if rgb_annot is not None and global_step % video_interval == 0:
                sim.render()
                rgb = rgb_annot.get_data()
                if rgb is not None and rgb.size > 0:
                    video_frames.append(rgb[:, :, :3].copy())

            elapsed += SIM_DT
            step_count += 1
            global_step += 1

            if step_count % log_every == 0:
                pos = drone.data.root_pos_w[0]
                vel = drone.data.root_lin_vel_w[0]
                quat = drone.data.root_quat_w[0]
                from DroneController import _quat_to_yaw
                cur_yaw = _quat_to_yaw(quat.unsqueeze(0))[0].item()
                yaw_err = abs(wrap_angle(controller._target_yaw.item() - cur_yaw))
                log(
                    f"  t={elapsed:.1f}s | "
                    f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) | "
                    f"yaw={math.degrees(cur_yaw):.1f} deg | "
                    f"dist={dist:.3f} m  yaw_err={math.degrees(yaw_err):.1f} deg"
                )

            quat = drone.data.root_quat_w[0]
            from DroneController import _quat_to_yaw
            cur_yaw = _quat_to_yaw(quat.unsqueeze(0))[0].item()
            yaw_err = abs(wrap_angle(controller._target_yaw.item() - cur_yaw))

            if dist < arrive_th and yaw_err < arrive_yaw_th:
                arrived = True
                log(
                    f"  REACHED  t={elapsed:.2f}s | "
                    f"dist={dist:.3f} m  yaw_err={math.degrees(yaw_err):.1f} deg"
                )
                break

        if not arrived:
            pos = drone.data.root_pos_w[0]
            log(
                f"  TIMEOUT  dist={dist:.3f} m | "
                f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
            )
            all_passed = False

        # -- Hover at waypoint ------------------------------------- #
        log(f"Hovering for {hover_secs:.1f} s ...")
        hover_elapsed = 0.0
        hover_log = max(1, int(hover_secs / 3 / SIM_DT))
        hover_step = 0

        while hover_elapsed < hover_secs:
            dist = controller.step(drone)
            drone.write_data_to_sim()
            sim.step()

            if controller.last_F_total is not None:
                avg_thrust = controller.last_F_total[0].item() / 4.0
                spin_rate = math.degrees(
                    ROTOR_SPEED_K * math.sqrt(max(avg_thrust, 0.0))
                )
                for i in range(4):
                    rotor_angles_deg[i] += rotor_dir[i] * spin_rate * SIM_DT
                    spin_ops[i].Set(float(rotor_angles_deg[i]))

            drone.update(SIM_DT)

            if rgb_annot is not None and global_step % video_interval == 0:
                sim.render()
                rgb = rgb_annot.get_data()
                if rgb is not None and rgb.size > 0:
                    video_frames.append(rgb[:, :, :3].copy())

            hover_elapsed += SIM_DT
            hover_step += 1
            global_step += 1

            if hover_step % hover_log == 0:
                pos = drone.data.root_pos_w[0]
                quat = drone.data.root_quat_w[0]
                from DroneController import _quat_to_yaw
                cur_yaw = _quat_to_yaw(quat.unsqueeze(0))[0].item()
                log(
                    f"  hover t={hover_elapsed:.1f}s | "
                    f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) | "
                    f"yaw={math.degrees(cur_yaw):.1f} deg | "
                    f"dist={dist:.3f} m"
                )

    # -- Final summary --------------------------------------------- #
    log(f"\n{'=' * 60}")
    pos = drone.data.root_pos_w[0]
    quat = drone.data.root_quat_w[0]
    from DroneController import _quat_to_yaw
    final_yaw = _quat_to_yaw(quat.unsqueeze(0))[0].item()
    status = "PASS" if all_passed else "PARTIAL FAIL"
    log(f"Result: {status}")
    log(
        f"Final:    pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  "
        f"yaw={math.degrees(final_yaw):.1f} deg"
    )
    log("Expected: pos=(0.000, 0.000, 1.500)  yaw=0.0 deg")

    # -- Save video ------------------------------------------------ #
    if video_frames:
        import cv2
        video_path = os.path.join(args_cli.video_dir, "drone_flight.mp4")
        h, w = video_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h)
        )
        for frame in video_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        log(
            f"Video saved: {video_path} "
            f"({len(video_frames)} frames, {len(video_frames) / 25.0:.1f} s)"
        )

    simulation_app.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
