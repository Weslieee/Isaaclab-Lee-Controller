"""Lee Position Controller for Quadrotor UAVs in NVIDIA Isaac Lab.

A geometric tracking controller based on:
    T. Lee, M. Leok, and N. H. McClamroch,
    "Geometric Tracking Control of a Quadrotor UAV on SE(3),"
    in Proc. IEEE CDC, 2010, pp. 5420-5425.

Implementation inspired by OmniDrones (btx0424/OmniDrones) and
ETH RotorS (ethz-asl/rotors_simulator).

Control Architecture (cascaded three-layer design)::

    Layer 1 - Position PD  ->  Desired thrust vector (world frame)
                |
    Layer 2 - Attitude PD  ->  Desired torque (body frame)
                |
    Layer 3 - Force/Torque ->  Applied to root body via Isaac Lab API

Requirements:
    - NVIDIA Isaac Lab (tested with Isaac Sim 6.0.0)
    - PyTorch
    - A quadrotor Articulation asset (tested with Crazyflie cf2x.usd)

Quick Start::

    from DroneController import DroneController

    # 1. Create controller
    controller = DroneController(device="cuda:0")

    # 2. Initialize after environment reset
    controller.reset(drone)  # drone: Isaac Lab Articulation

    # 3. Set target (position delta in world frame)
    controller.set_target_delta(dx=2.0, dy=0.0, dz=0.5)

    # 4. Step every physics tick
    for step in range(num_steps):
        dist = controller.step(drone, dt=0.005)
        drone.write_data_to_sim()
        sim.step()
        drone.update(dt)

Integration with VLA Models::

    # VLA inference loop
    action = vla_model.predict(image, instruction)
    dx, dy, dz, dyaw = action[0], action[1], action[2], action[3]
    controller.set_target_delta(dx, dy, dz, dyaw)

License:
    BSD-3-Clause
"""

import math

import torch


class DroneController:
    """Lee geometric position controller for quadrotor UAVs.

    This controller implements a cascaded PD control scheme:

    - **Outer loop**: Position PD controller computes a desired thrust
      vector in the world frame, including gravity compensation.
    - **Inner loop**: Attitude PD controller computes the desired torque
      in the body frame to align the vehicle's z-axis with the desired
      thrust direction.
    - **Actuation**: World-frame force and body-frame torque are applied
      to the root body via ``set_external_force_and_torque``.

    The controller is stateless across physics steps (no hidden integrators
    that accumulate without bound), making it robust to long-horizon tasks.

    All computations use PyTorch tensors and support batched environments
    on GPU for efficient parallel simulation.

    Args:
        device: Torch device string (e.g., ``"cuda:0"`` or ``"cpu"``).
        gravity: Gravitational acceleration (m/s^2). Default: ``9.81``.
        arm_length: Rotor arm length (m). Default: ``0.046`` (Crazyflie).
        km_kf_ratio: Torque-to-thrust coefficient ratio for yaw control.
            Default: ``0.006``.
        Ixx: Moment of inertia about x-axis (kg*m^2). Default: ``1.4e-5``.
        Iyy: Moment of inertia about y-axis (kg*m^2). Default: ``1.4e-5``.
        Izz: Moment of inertia about z-axis (kg*m^2). Default: ``2.17e-5``.
        Kp_xy: Position proportional gain for x/y axes. Default: ``2.0``.
        Kd_xy: Velocity damping gain for x/y axes. Default: ``2.5``.
        Kp_z: Position proportional gain for z axis. Default: ``3.0``.
        Kd_z: Velocity damping gain for z axis. Default: ``3.0``.
        KR_xy: Rotation error gain for roll/pitch (N*m). Default: ``8e-4``.
        KR_z: Rotation error gain for yaw (N*m). Default: ``4e-4``.
        Kw_xy: Angular velocity gain for roll/pitch (N*m*s). Default: ``2e-4``.
        Kw_z: Angular velocity gain for yaw (N*m*s). Default: ``1e-4``.
        max_tilt_angle: Maximum tilt angle (rad). Default: ``0.5`` (~28 deg).
        min_thrust: Minimum per-rotor thrust (N). Default: ``0.0``.
        max_thrust_ratio: Maximum total thrust as a multiple of hover
            thrust. Default: ``3.0``.

    Example::

        controller = DroneController(device="cuda:0", Kp_xy=3.0)
        controller.reset(drone)
        controller.set_target_delta(1.0, 0.0, 0.5)
        dist = controller.step(drone, dt=0.005)
        print(f"Distance to target: {dist:.3f} m")
    """

    def __init__(
        self,
        device: str = "cuda:0",
        gravity: float = 9.81,
        # Crazyflie physical parameters
        arm_length: float = 0.046,
        km_kf_ratio: float = 0.006,
        # Inertia tensor (Crazyflie, kg*m^2)
        Ixx: float = 1.4e-5,
        Iyy: float = 1.4e-5,
        Izz: float = 2.17e-5,
        # Position PD gains
        Kp_xy: float = 2.0,
        Kd_xy: float = 2.5,
        Kp_z: float = 3.0,
        Kd_z: float = 3.0,
        # Attitude PD gains
        KR_xy: float = 8e-4,
        KR_z: float = 4e-4,
        Kw_xy: float = 2e-4,
        Kw_z: float = 1e-4,
        # Safety limits
        max_tilt_angle: float = 0.5,
        min_thrust: float = 0.0,
        max_thrust_ratio: float = 3.0,
    ) -> None:
        self.device = device
        self.g = gravity
        self.arm = arm_length
        self.km_kf = km_kf_ratio

        self._inertia = torch.diag(
            torch.tensor([Ixx, Iyy, Izz], device=device)
        )

        self._Kp = torch.tensor([Kp_xy, Kp_xy, Kp_z], device=device)
        self._Kd = torch.tensor([Kd_xy, Kd_xy, Kd_z], device=device)
        self._KR = torch.tensor([KR_xy, KR_xy, KR_z], device=device)
        self._Kw = torch.tensor([Kw_xy, Kw_xy, Kw_z], device=device)

        self._max_tilt = max_tilt_angle
        self._min_thrust = min_thrust
        self._max_thrust_ratio = max_thrust_ratio

        self._target_pos = None
        self._target_yaw = None
        self._mass = None
        self._max_total_thrust = 0.0
        self._initialized = False

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def reset(self, drone) -> None:
        """Initialize controller state from the current drone pose.

        Reads the true mass from PhysX so that the gravity feedforward
        term is always accurate. Sets the initial target to the current
        position and yaw (i.e., hover in place).

        Must be called once after every ``env.reset()`` and before the
        first call to :meth:`step`.

        Args:
            drone: An Isaac Lab ``Articulation`` object.
        """
        self._mass = drone.root_physx_view.get_masses()[0].sum()
        hover_per_rotor = self._mass * self.g / 4.0
        self._max_total_thrust = hover_per_rotor * 4.0 * self._max_thrust_ratio

        self._target_pos = drone.data.root_pos_w[0:1].clone()

        quat = drone.data.root_quat_w[0]
        self._target_yaw = _quat_to_yaw(quat.unsqueeze(0))[0]

        self._initialized = True

        print(
            f"[DroneController] reset | "
            f"mass={self._mass.item():.4f} kg | "
            f"hover={hover_per_rotor.item():.5f} N/rotor | "
            f"pos={self._target_pos[0].tolist()}",
            flush=True,
        )

    def set_target_delta(
        self, dx: float, dy: float, dz: float, dyaw: float = 0.0
    ) -> None:
        """Update the target by a position/yaw delta in the world frame.

        Each call **accumulates** onto the previous target (i.e., targets
        are chained, not relative to the current drone position).

        Args:
            dx: Position delta along world X (m).
            dy: Position delta along world Y (m).
            dz: Position delta along world Z (m).
            dyaw: Yaw delta (rad). Positive = counter-clockwise viewed
                from above. Default: ``0.0``.
        """
        self._check_init()
        self._target_pos = self._target_pos + torch.tensor(
            [[dx, dy, dz]], device=self.device
        )
        self._target_yaw = self._target_yaw + dyaw
        print(
            f"[DroneController] target -> "
            f"pos={self._target_pos[0].tolist()} "
            f"yaw={math.degrees(self._target_yaw.item()):.1f} deg "
            f"(delta: {dx:.3f}, {dy:.3f}, {dz:.3f}, "
            f"{math.degrees(dyaw):.1f} deg)",
            flush=True,
        )

    def set_target_absolute(self, x: float, y: float, z: float) -> None:
        """Set an absolute target position in the world frame.

        Args:
            x: World X coordinate (m).
            y: World Y coordinate (m).
            z: World Z coordinate (m).
        """
        self._check_init()
        self._target_pos = torch.tensor([[x, y, z]], device=self.device)

    def step(self, drone, dt: float = 0.005) -> float:
        """Run one control step and apply forces/torques to the drone.

        Must be called **before** ``sim.step()`` in the simulation loop.

        Args:
            drone: An Isaac Lab ``Articulation`` object.
            dt: Physics timestep in seconds. Default: ``0.005`` (200 Hz).

        Returns:
            Euclidean distance from the drone to the target (m).
        """
        self._check_init()
        num_envs = drone.data.root_pos_w.shape[0]

        # -- Read state ------------------------------------------------ #
        pos = drone.data.root_pos_w          # (N, 3)
        vel = drone.data.root_lin_vel_w      # (N, 3)
        quat = drone.data.root_quat_w       # (N, 4) [w, x, y, z]
        omega = drone.data.root_ang_vel_b    # (N, 3) body frame

        R = _quat_to_rotmat(quat)            # (N, 3, 3) body -> world

        # -- Layer 1: Position PD -> desired force (world frame) ------- #
        target = self._target_pos.expand(num_envs, 3)
        e_pos = target - pos
        e_vel = -vel

        accel_des = self._Kp * e_pos + self._Kd * e_vel
        F_des = self._mass * accel_des
        F_des[:, 2] += self._mass * self.g   # gravity feedforward

        # Enforce tilt limit
        F_xy_max = F_des[:, 2] * math.tan(self._max_tilt)
        F_xy_norm = F_des[:, :2].norm(dim=-1, keepdim=True)
        scale = (F_xy_max.unsqueeze(-1) / (F_xy_norm + 1e-8)).clamp(max=1.0)
        F_des[:, :2] *= scale

        # Clamp total thrust magnitude
        F_mag = F_des.norm(dim=-1, keepdim=True)
        thrust_scale = (
            self._max_total_thrust / (F_mag + 1e-8)
        ).clamp(max=1.0)
        F_des = F_des * thrust_scale

        # -- Layer 2: Attitude PD -> desired torque (body frame) ------- #
        b3_des = F_des / (F_des.norm(dim=-1, keepdim=True) + 1e-8)
        R_des = _desired_rotmat(b3_des, self._target_yaw.expand(num_envs))

        e_R = _rotation_error(R, R_des)
        e_w = omega

        I_omega = (
            self._inertia.unsqueeze(0) @ omega.unsqueeze(-1)
        ).squeeze(-1)
        gyroscopic = torch.cross(omega, I_omega, dim=-1)
        M = -self._KR * e_R - self._Kw * e_w + gyroscopic

        # -- Layer 3: Apply to root body ------------------------------- #
        num_bodies = drone.num_bodies
        forces = torch.zeros(num_envs, num_bodies, 3, device=self.device)
        torques = torch.zeros(num_envs, num_bodies, 3, device=self.device)
        forces[:, 0, :] = F_des   # world-frame force
        torques[:, 0, :] = M      # body-frame torque

        drone.set_external_force_and_torque(forces, torques)

        return e_pos.norm(dim=-1).mean().item()

    # ------------------------------------------------------------------ #
    #  Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def target(self) -> torch.Tensor:
        """Current target position as a (1, 3) tensor (world frame)."""
        return self._target_pos

    @property
    def target_yaw_deg(self) -> float:
        """Current target yaw angle in degrees."""
        if self._target_yaw is None:
            return 0.0
        return math.degrees(self._target_yaw.item())

    @property
    def drone_mass(self) -> float:
        """Drone mass in kg (read from PhysX at reset)."""
        return self._mass.item() if self._mass is not None else 0.0

    # ------------------------------------------------------------------ #
    #  Internal                                                           #
    # ------------------------------------------------------------------ #

    def _check_init(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "DroneController not initialized. Call reset(drone) first."
            )


# ====================================================================== #
#  Stateless helper functions (module-level, pure math)                   #
# ====================================================================== #


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Quaternion [w,x,y,z] -> rotation matrix (body -> world).

    Args:
        q: Shape (N, 4).

    Returns:
        Shape (N, 3, 3).
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = torch.zeros(N, 3, 3, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from quaternion [w,x,y,z].

    Args:
        q: Shape (N, 4).

    Returns:
        Shape (N,), radians.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _desired_rotmat(
    b3_des: torch.Tensor, yaw_des: torch.Tensor
) -> torch.Tensor:
    """Build desired rotation matrix from thrust direction and yaw.

    Args:
        b3_des: Desired body z-axis (unit), shape (N, 3).
        yaw_des: Desired yaw angles, shape (N,), radians.

    Returns:
        Shape (N, 3, 3).
    """
    N = b3_des.shape[0]
    device = b3_des.device

    c_yaw = torch.cos(yaw_des)
    s_yaw = torch.sin(yaw_des)
    heading = torch.stack(
        [c_yaw, s_yaw, torch.zeros(N, device=device)], dim=-1
    )

    b2_des = torch.cross(b3_des, heading, dim=-1)
    b2_norm = b2_des.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b2_des = b2_des / b2_norm

    b1_des = torch.cross(b2_des, b3_des, dim=-1)

    return torch.stack([b1_des, b2_des, b3_des], dim=-1)


def _rotation_error(
    R: torch.Tensor, R_des: torch.Tensor
) -> torch.Tensor:
    """Lee rotation error: e_R = 0.5 * vee(R_des^T R - R^T R_des).

    Args:
        R: Current rotation matrices, shape (N, 3, 3).
        R_des: Desired rotation matrices, shape (N, 3, 3).

    Returns:
        Rotation error vectors, shape (N, 3).
    """
    eR_mat = R_des.transpose(-1, -2) @ R - R.transpose(-1, -2) @ R_des
    return 0.5 * torch.stack(
        [
            eR_mat[:, 2, 1] - eR_mat[:, 1, 2],
            eR_mat[:, 0, 2] - eR_mat[:, 2, 0],
            eR_mat[:, 1, 0] - eR_mat[:, 0, 1],
        ],
        dim=-1,
    )
