"""Lee Position Controller for Quadrotor UAVs in NVIDIA Isaac Lab.

A geometric tracking controller based on:
    T. Lee, M. Leok, and N. H. McClamroch,
    "Geometric Tracking Control of a Quadrotor UAV on SE(3),"
    in Proc. IEEE CDC, 2010, pp. 5420-5425.

Implementation inspired by OmniDrones (btx0424/OmniDrones) and
ETH RotorS (ethz-asl/rotors_simulator).

Control Architecture (cascaded three-layer design)::

    Layer 1 - Position PD       ->  Desired thrust vector  (world frame)
                |
    Layer 2 - Attitude PD (Lee) ->  Desired torque         (body frame)
                |
    Layer 3 - Actuation         ->  Thrust along body +z + torque applied
                                    via Isaac Lab ``set_external_force_and_torque``

A real quadrotor can only push along its body +z axis. The controller therefore
converts the world-frame desired force into a scalar thrust ``T = F_des . b3``,
and the horizontal motion arises naturally from the attitude loop tilting the
body. This avoids the common pitfall of applying arbitrary world-frame forces,
which would let the drone translate without any visible tilt.

Requirements:
    - NVIDIA Isaac Lab (tested with Isaac Sim 6.0.0)
    - PyTorch
    - A quadrotor Articulation asset (tested with Crazyflie cf2x.usd)

Quick Start::

    from DroneController import DroneController

    # 1. Create controller (default gains tuned for Crazyflie)
    controller = DroneController(device="cuda:0")

    # 2. Initialize after every env reset
    controller.reset(drone)  # drone: Isaac Lab Articulation

    # 3. Update target from VLA output (world-frame delta)
    controller.set_target_delta(dx=2.0, dy=0.0, dz=0.5, dyaw=0.0)

    # 4. Step every physics tick, BEFORE sim.step()
    for _ in range(num_steps):
        dist = controller.step(drone)
        drone.write_data_to_sim()
        sim.step()
        drone.update(SIM_DT)

Integration with VLA Models::

    # High-level VLA + controller loop
    for step in range(total_steps):
        if step % vla_interval == 0:
            image = get_camera_image()
            dx, dy, dz, dyaw = vla_model.predict(image, instruction)
            controller.set_target_delta(dx, dy, dz, dyaw)
        controller.step(drone)
        drone.write_data_to_sim()
        sim.step()
        drone.update(SIM_DT)

License:
    BSD-3-Clause
"""

from __future__ import annotations

import math

import torch


class DroneController:
    """Lee geometric position controller for quadrotor UAVs.

    The controller implements a cascaded PD scheme on SE(3):

    - **Outer loop**: Position PD computes a desired force vector in the
      world frame, including gravity feedforward.
    - **Inner loop**: Attitude PD computes a desired body-frame torque
      that aligns the body z-axis with the desired thrust direction
      while tracking a desired yaw.
    - **Actuation**: Only the projection of the desired force onto the
      *current* body z-axis is applied as thrust (``T = F_des . b3``),
      ensuring physically realistic tilt-to-translate behavior.

    The controller is stateless across physics steps (no integrators
    that grow without bound), making it robust for long-horizon RL or
    VLA-driven navigation tasks.

    All computations use PyTorch tensors and support batched
    environments on GPU.

    Args:
        device: Torch device string (e.g., ``"cuda:0"`` or ``"cpu"``).
        gravity: Gravitational acceleration (m/s^2). Default: ``9.81``.
        arm_length: Rotor arm length (m). Reserved for future motor
            allocation; unused in the current force/torque-level
            implementation. Default: ``0.046`` (Crazyflie).
        km_kf_ratio: Torque-to-thrust coefficient ratio. Reserved for
            future motor allocation. Default: ``0.006``.
        Ixx: Moment of inertia about body x-axis (kg*m^2).
            Default: ``1.4e-5``.
        Iyy: Moment of inertia about body y-axis (kg*m^2).
            Default: ``1.4e-5``.
        Izz: Moment of inertia about body z-axis (kg*m^2).
            Default: ``2.17e-5``.
        Kp_xy: Position proportional gain for x/y axes. Default: ``1.2``.
        Kd_xy: Velocity damping gain for x/y axes. Default: ``1.8``.
        Kp_z: Position proportional gain for z axis. Default: ``2.0``.
        Kd_z: Velocity damping gain for z axis. Default: ``2.5``.
        KR_xy: Rotation error gain for roll/pitch. Default: ``8e-4``.
        KR_z: Rotation error gain for yaw. Default: ``4e-4``.
        Kw_xy: Angular velocity gain for roll/pitch. Default: ``2e-4``.
        Kw_z: Angular velocity gain for yaw. Default: ``1e-4``.
        max_tilt_angle: Maximum tilt angle (rad). Default: ``0.35``
            (~20 deg).
        max_thrust_ratio: Maximum total thrust as a multiple of hover
            thrust. Default: ``3.0``.
        min_thrust_ratio: Minimum vertical component of the desired
            force, expressed as a fraction of ``mass * gravity``. Acts
            as a safety floor that prevents the desired body z-axis
            from flipping. Default: ``0.1``.
        verbose: If ``True``, print target updates and reset banners.
            Default: ``False``.

    Example::

        controller = DroneController(device="cuda:0", verbose=True)
        controller.reset(drone)
        controller.set_target_delta(1.0, 0.0, 0.5)
        dist = controller.step(drone)
    """

    def __init__(
        self,
        device: str = "cuda:0",
        gravity: float = 9.81,
        # Crazyflie physical parameters (reserved for motor allocation)
        arm_length: float = 0.046,
        km_kf_ratio: float = 0.006,
        # Inertia tensor (Crazyflie, kg*m^2)
        Ixx: float = 1.4e-5,
        Iyy: float = 1.4e-5,
        Izz: float = 2.17e-5,
        # Position PD gains
        Kp_xy: float = 1.2,
        Kd_xy: float = 1.8,
        Kp_z: float = 2.0,
        Kd_z: float = 2.5,
        # Attitude PD gains
        KR_xy: float = 8e-4,
        KR_z: float = 4e-4,
        Kw_xy: float = 2e-4,
        Kw_z: float = 1e-4,
        # Safety limits
        max_tilt_angle: float = 0.35,
        max_thrust_ratio: float = 3.0,
        min_thrust_ratio: float = 0.1,
        # Logging
        verbose: bool = False,
    ) -> None:
        self.device = device
        self.g = gravity
        self.arm = arm_length
        self.km_kf = km_kf_ratio
        self.verbose = verbose

        self._inertia = torch.diag(
            torch.tensor([Ixx, Iyy, Izz], device=device)
        )

        self._Kp = torch.tensor([Kp_xy, Kp_xy, Kp_z], device=device)
        self._Kd = torch.tensor([Kd_xy, Kd_xy, Kd_z], device=device)
        self._KR = torch.tensor([KR_xy, KR_xy, KR_z], device=device)
        self._Kw = torch.tensor([Kw_xy, Kw_xy, Kw_z], device=device)

        self._max_tilt = max_tilt_angle
        self._max_thrust_ratio = max_thrust_ratio
        self._min_thrust_ratio = min_thrust_ratio

        # Runtime state (initialized in reset)
        self._target_pos: torch.Tensor | None = None       # (1, 3)
        self._target_yaw: torch.Tensor | None = None       # (1,) rad in (-pi, pi]
        self._mass: torch.Tensor | None = None             # scalar
        self._hover_thrust: torch.Tensor | None = None     # per-rotor (N)
        self._max_total_thrust: torch.Tensor | None = None
        self._min_thrust_z: torch.Tensor | None = None
        self._last_thrust: torch.Tensor | None = None      # (N,)
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
        self._mass = drone.root_physx_view.get_masses()[0].sum().to(self.device)
        self._hover_thrust = self._mass * self.g / 4.0
        self._max_total_thrust = self._hover_thrust * 4.0 * self._max_thrust_ratio
        self._min_thrust_z = self._mass * self.g * self._min_thrust_ratio

        self._target_pos = drone.data.root_pos_w[0:1].clone()
        self._target_yaw = _wrap_to_pi(_quat_to_yaw(drone.data.root_quat_w[0:1]))

        self._initialized = True

        if self.verbose:
            print(
                f"[DroneController] reset | "
                f"mass={self._mass.item():.4f} kg | "
                f"hover={self._hover_thrust.item():.5f} N/rotor | "
                f"pos={self._target_pos[0].tolist()} | "
                f"yaw={math.degrees(self._target_yaw.item()):.1f} deg",
                flush=True,
            )

    def set_target_delta(
        self, dx: float, dy: float, dz: float, dyaw: float = 0.0
    ) -> None:
        """Update the target by a position/yaw delta in the world frame.

        Each call **accumulates** onto the previous target. Yaw is
        wrapped to (-pi, pi] after every update.

        Args:
            dx: Position delta along world X (m).
            dy: Position delta along world Y (m).
            dz: Position delta along world Z (m).
            dyaw: Yaw delta (rad). Positive = counter-clockwise viewed
                from above. Default: ``0.0``.
        """
        self._check_init()
        delta = torch.tensor([[dx, dy, dz]], device=self.device)
        self._target_pos = self._target_pos + delta
        self._target_yaw = _wrap_to_pi(self._target_yaw + dyaw)

        if self.verbose:
            print(
                f"[DroneController] target -> "
                f"pos={self._target_pos[0].tolist()} "
                f"yaw={math.degrees(self._target_yaw.item()):.1f} deg "
                f"(delta: {dx:.3f}, {dy:.3f}, {dz:.3f}, "
                f"{math.degrees(dyaw):.1f} deg)",
                flush=True,
            )

    def set_target_absolute(
        self, x: float, y: float, z: float, yaw: float | None = None
    ) -> None:
        """Set an absolute target position in the world frame.

        Args:
            x: World X coordinate (m).
            y: World Y coordinate (m).
            z: World Z coordinate (m).
            yaw: Optional absolute yaw target (rad). If ``None``, the
                current target yaw is preserved. Default: ``None``.
        """
        self._check_init()
        self._target_pos = torch.tensor([[x, y, z]], device=self.device)
        if yaw is not None:
            self._target_yaw = _wrap_to_pi(
                torch.tensor([yaw], device=self.device, dtype=self._target_yaw.dtype)
            )

    def yaw_error(self, drone) -> torch.Tensor:
        """Return the shortest yaw error per environment.

        Args:
            drone: An Isaac Lab ``Articulation`` object.

        Returns:
            Tensor of shape ``(N,)`` with values in ``[-pi, pi]``.
        """
        self._check_init()
        cur_yaw = _quat_to_yaw(drone.data.root_quat_w)
        return _wrap_to_pi(self._target_yaw.expand_as(cur_yaw) - cur_yaw)

    def step(self, drone, dt: float = 0.005) -> float:
        """Run one control step and apply thrust/torque to the drone.

        Must be called **before** ``sim.step()`` in the simulation loop.

        Args:
            drone: An Isaac Lab ``Articulation`` object.
            dt: Physics timestep (s). Currently unused (the controller
                is stateless across steps); kept for API stability.
                Default: ``0.005`` (200 Hz).

        Returns:
            Mean Euclidean distance from drone to target (m).
        """
        self._check_init()
        del dt  # currently unused; kept for API stability
        num_envs = drone.data.root_pos_w.shape[0]

        # -- Read state ------------------------------------------------ #
        pos = drone.data.root_pos_w           # (N, 3)
        vel = drone.data.root_lin_vel_w       # (N, 3)
        quat = drone.data.root_quat_w        # (N, 4) [w, x, y, z]
        omega = drone.data.root_ang_vel_b     # (N, 3) body frame

        R = _quat_to_rotmat(quat)             # (N, 3, 3) body -> world

        # -- Layer 1: Position PD -> desired force (world frame) ------- #
        target = self._target_pos.expand(num_envs, 3)
        e_pos = target - pos
        e_vel = -vel

        accel_des = self._Kp * e_pos + self._Kd * e_vel
        F_des = self._mass * accel_des
        F_des[:, 2] = F_des[:, 2] + self._mass * self.g  # gravity feedforward

        # Floor on vertical force component (avoid attitude flip when
        # the drone overshoots above the target with high downward gain)
        F_des[:, 2] = F_des[:, 2].clamp(min=self._min_thrust_z.item())

        # Tilt limit: |F_xy| <= F_z * tan(max_tilt)
        F_xy_max = (F_des[:, 2] * math.tan(self._max_tilt)).unsqueeze(-1)
        F_xy_norm = F_des[:, :2].norm(dim=-1, keepdim=True)
        scale_xy = (F_xy_max / (F_xy_norm + 1e-8)).clamp(max=1.0)
        F_des[:, :2] = F_des[:, :2] * scale_xy

        # Total thrust magnitude limit
        F_mag = F_des.norm(dim=-1, keepdim=True)
        scale_total = (
            self._max_total_thrust / (F_mag + 1e-8)
        ).clamp(max=1.0)
        F_des = F_des * scale_total

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

        # -- Layer 3: Apply thrust along body z + torque --------------- #
        # A real quadrotor can only push along body +z. The horizontal
        # motion must come from the attitude loop tilting the body.
        # Project F_des onto the actual body z-axis to obtain the scalar
        # thrust, then apply it along body +z (body frame).
        b3_actual = R[:, :, 2]                          # (N, 3)
        thrust = (F_des * b3_actual).sum(dim=-1).clamp(
            min=0.0, max=self._max_total_thrust.item()
        )                                               # (N,)

        body_force = torch.zeros(num_envs, 3, device=self.device)
        body_force[:, 2] = thrust

        num_bodies = drone.num_bodies
        forces = torch.zeros(num_envs, num_bodies, 3, device=self.device)
        torques = torch.zeros(num_envs, num_bodies, 3, device=self.device)
        forces[:, 0, :] = body_force    # body-frame thrust on root link
        torques[:, 0, :] = M            # body-frame torque on root link

        drone.set_external_force_and_torque(forces, torques)

        self._last_thrust = thrust.detach().clone()
        return e_pos.norm(dim=-1).mean().item()

    # ------------------------------------------------------------------ #
    #  Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def target(self) -> torch.Tensor:
        """Current target position as a (1, 3) tensor (world frame)."""
        return self._target_pos

    @property
    def target_yaw(self) -> torch.Tensor:
        """Current target yaw as a (1,) tensor (radians, wrapped)."""
        return self._target_yaw

    @property
    def target_yaw_deg(self) -> float:
        """Current target yaw in degrees."""
        if self._target_yaw is None:
            return 0.0
        return math.degrees(self._target_yaw.item())

    @property
    def drone_mass(self) -> float:
        """Drone mass in kg (read from PhysX at reset)."""
        return self._mass.item() if self._mass is not None else 0.0

    @property
    def last_thrust(self) -> torch.Tensor | None:
        """Total thrust applied at the previous :meth:`step` call.

        Shape ``(N,)`` in Newtons. Useful for visualizing rotor spin.
        Returns ``None`` before the first step.
        """
        return self._last_thrust

    # Backward-compatible alias for downstream scripts.
    @property
    def last_F_total(self) -> torch.Tensor | None:
        """Alias of :attr:`last_thrust` (kept for backward compatibility)."""
        return self._last_thrust

    # ------------------------------------------------------------------ #
    #  Internal                                                           #
    # ------------------------------------------------------------------ #

    def _check_init(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "DroneController not initialized. Call reset(drone) first."
            )


# ====================================================================== #
#  Module-level math helpers                                              #
# ====================================================================== #

def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to ``(-pi, pi]`` element-wise."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Quaternion ``[w, x, y, z]`` -> rotation matrix (body -> world).

    Args:
        q: Shape ``(N, 4)``.

    Returns:
        Shape ``(N, 3, 3)``.
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
    """Extract yaw (rotation about world z) from quaternion ``[w, x, y, z]``.

    Args:
        q: Shape ``(N, 4)``.

    Returns:
        Shape ``(N,)``, radians.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _desired_rotmat(
    b3_des: torch.Tensor, yaw_des: torch.Tensor
) -> torch.Tensor:
    """Build desired rotation matrix from thrust direction and yaw.

    Uses the standard Lee construction: the desired heading vector
    ``x_C = (cos yaw, sin yaw, 0)`` is projected onto the plane normal
    to the desired body z-axis. The degenerate case where ``b3_des``
    aligns with ``x_C`` falls back to a 90-degree-rotated heading.

    Args:
        b3_des: Desired body z-axis (unit), shape ``(N, 3)``.
        yaw_des: Desired yaw angles, shape ``(N,)``, radians.

    Returns:
        Rotation matrices, shape ``(N, 3, 3)``.
    """
    cos_yaw = torch.cos(yaw_des)
    sin_yaw = torch.sin(yaw_des)
    zero = torch.zeros_like(cos_yaw)
    x_c = torch.stack([cos_yaw, sin_yaw, zero], dim=-1)

    b2 = torch.cross(b3_des, x_c, dim=-1)
    b2_norm = b2.norm(dim=-1, keepdim=True)
    degenerate = (b2_norm < 1e-6).squeeze(-1)
    if degenerate.any():
        y_c = torch.stack([-sin_yaw, cos_yaw, zero], dim=-1)
        b2_alt = torch.cross(b3_des, y_c, dim=-1)
        b2 = torch.where(degenerate.unsqueeze(-1), b2_alt, b2)
        b2_norm = b2.norm(dim=-1, keepdim=True)
    b2 = b2 / b2_norm.clamp(min=1e-8)

    b1 = torch.cross(b2, b3_des, dim=-1)
    return torch.stack([b1, b2, b3_des], dim=-1)


def _rotation_error(
    R: torch.Tensor, R_des: torch.Tensor
) -> torch.Tensor:
    """Lee rotation error ``e_R = 0.5 * vee(R_des^T R - R^T R_des)``.

    Args:
        R: Current rotation matrices, shape ``(N, 3, 3)``.
        R_des: Desired rotation matrices, shape ``(N, 3, 3)``.

    Returns:
        Rotation error vectors, shape ``(N, 3)``.
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
