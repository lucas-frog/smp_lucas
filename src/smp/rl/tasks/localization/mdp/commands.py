from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)
    self.is_world_env = torch.zeros_like(self.is_heading_env)
    self.is_forward_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    # Randomly assign world-frame envs.
    self.is_world_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_world_envs
    # Copy sampled velocities as world-frame reference for world envs.
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    # Forward-only envs: positive lin_vel_x, zero lateral and angular.
    self.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
    fwd_ids = env_ids[self.is_forward_env[env_ids]]
    if len(fwd_ids) > 0:
      self.vel_command_b[fwd_ids, 0] = (
        self.vel_command_b[fwd_ids, 0].abs().clamp(min=0.3)
      )
      self.vel_command_b[fwd_ids, 1] = 0.0
      self.vel_command_b[fwd_ids, 2] = 0.0

    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask]
    if len(init_vel_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    # World-frame envs: rotate world-frame linear vel into body frame.
    if self.is_world_env.any():
      w_ids = self.is_world_env.nonzero(as_tuple=False).flatten()
      heading = self.robot.data.heading_w[w_ids]
      cos_h = torch.cos(heading)
      sin_h = torch.sin(heading)
      vx_w = self.vel_command_w[w_ids, 0]
      vy_w = self.vel_command_w[w_ids, 1]
      self.vel_command_b[w_ids, 0] = cos_h * vx_w + sin_h * vy_w
      self.vel_command_b[w_ids, 1] = -sin_h * vx_w + cos_h * vy_w

    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0
    self.vel_command_w[standing_env_ids, :] = 0.0

  # GUI.

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
    on_change: Callable[..., None] | None = None,
    request_action: Callable[..., None] | None = None,
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges

    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        max_input = server.gui.add_slider(
            f"Max {label}",
            initial_value=max_val,
            step=0.1,
            min=0.0,
            max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Helper to transform local to world coordinates.
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity arrow (blue).
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan).
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


class GSIResetVelocityCommand(CommandTerm):
  """Command that mirrors the GSI-reset root velocity and holds it constant.

  At episode reset the ``gsi_reset`` event writes a motion-prior sample into
  the simulator.  This command reads the resulting body-frame root velocity
  (linear xy + angular z) on the first step and uses it as a fixed command
  for the rest of the episode — no resampling, no standing/heading/world
  modes, just "keep doing what GSI started."

  The command format ``[vx, vy, ω_z]`` in the robot's body frame matches
  ``UniformVelocityCommand``, so downstream observations, rewards, and
  deployment joystick mapping stay unchanged.
  """

  cfg: GSIResetVelocityCommandCfg

  def __init__(self, cfg: GSIResetVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    # Body-frame twist command: [vx, vy, ω_z].
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

    # Per-env flag: True → capture sim velocity on next _update_command().
    self._needs_capture = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # GUI (same pattern as UniformVelocityCommand).
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_step = self._env.cfg.episode_length_s / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Mark envs for GSI-velocity capture on the next ``_update_command``."""
    self._needs_capture[env_ids] = True

  def _update_command(self) -> None:
    """Capture body-frame root velocity from sim for newly-reset envs."""
    if self._needs_capture.any():
      ids = self._needs_capture.nonzero(as_tuple=False).flatten()
      self.vel_command_b[ids, 0] = self.robot.data.root_link_lin_vel_b[ids, 0]
      self.vel_command_b[ids, 1] = self.robot.data.root_link_lin_vel_b[ids, 1]
      self.vel_command_b[ids, 2] = self.robot.data.root_link_ang_vel_b[ids, 2]
      self._needs_capture[ids] = False

  # ── GUI (same as UniformVelocityCommand) ──────────────────────────────

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
    on_change: Callable[..., None] | None = None,
    request_action: Callable[..., None] | None = None,
  ) -> None:
    from viser import Icon

    r = self.cfg.ranges
    axes = [
      ("lin_vel_x", r.lin_vel_x[1]),
      ("lin_vel_y", r.lin_vel_y[1]),
      ("ang_vel_z", r.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}", initial_value=max_val, step=0.1, min=0.0, max=10.0,
        )
        slider = server.gui.add_slider(
          label, min=-max_val, max=max_val, step=0.05, initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value
        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    """Step the command: capture GSI velocity on first call after reset, then
    hold constant.  Skips the parent's resampling timer entirely."""
    self._update_metrics()
    self._update_command()
    # GUI override.
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value

  # ── Debug visualization (same as UniformVelocityCommand) ──────────────

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity (blue).
      o = local_to_world(np.array([0, 0, z_offset]) * scale)
      visualizer.add_arrow(
        o, local_to_world((np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale),
        color=(0.2, 0.2, 0.6, 0.6), width=0.015,
      )
      # Command angular velocity (green).
      visualizer.add_arrow(
        o, local_to_world((np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale),
        color=(0.2, 0.6, 0.2, 0.6), width=0.015,
      )
      # Actual linear velocity (cyan).
      visualizer.add_arrow(
        o, local_to_world((np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale),
        color=(0.0, 0.6, 1.0, 0.7), width=0.015,
      )
      # Actual angular velocity (light green).
      visualizer.add_arrow(
        o, local_to_world((np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale),
        color=(0.0, 1.0, 0.4, 0.7), width=0.015,
      )


@dataclass(kw_only=True)
class GSIResetVelocityCommandCfg(CommandTermCfg):
  """Configuration for ``GSIResetVelocityCommand``.

  ``ranges`` only serves as GUI slider limits (the actual command values come
  from the GSI-reset velocity).  ``resampling_time_range`` is unused for
  rescheduling but kept to satisfy the parent signature.
  """

  entity_name: str

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float] = (-3.0, 5.0)
    lin_vel_y: tuple[float, float] = (-3.0, 3.0)
    ang_vel_z: tuple[float, float] = (-2.0, 2.0)

  ranges: Ranges = field(default_factory=Ranges)

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> GSIResetVelocityCommand:
    return GSIResetVelocityCommand(self, env)


class DirectionalVelocityCommand(CommandTerm):
  """Per-reset directional velocity command — no periodic resampling.

  At episode reset each environment is assigned to one of five linear categories:
  forward (+x), backward (-x), left (+y), right (-y), or standing.  Angular
  velocity ``ω_z`` is then independently overlaid onto forward, backward, and
  standing environments with configurable probability, so the robot walks a
  curve or learns to turn in place.

  The sampled body-frame twist ``[vx, vy, ω_z]`` is held constant until the next
  reset.
  """

  cfg: DirectionalVelocityCommandCfg

  # Category indices (stored as ints for fast indexing).
  CAT_FWD = 0   # +x
  CAT_BWD = 1   # -x
  CAT_LEFT = 2  # +y
  CAT_RIGHT = 3 # -y
  CAT_STAND = 4

  def __init__(self, cfg: DirectionalVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self._category = torch.full(
      (self.num_envs,), self.CAT_STAND, dtype=torch.long, device=self.device
    )

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # GUI (same pattern as GSIReset).
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  @property
  def is_standing_env(self) -> torch.Tensor:
    """Exposed for ``task_smp_product`` SMP-gate bypass (same API as
    ``UniformVelocityCommand``)."""
    return self._category == self.CAT_STAND

  def _update_metrics(self) -> None:
    max_step = self._env.cfg.episode_length_s / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Assign each env a linear category, then overlay angular on fwd/bwd/standing."""
    n = int(env_ids.numel())
    if n == 0:
      return
    r = torch.rand(n, device=self.device)

    # ── Linear / standing category (sums to 1.0) ──────────────────────────
    probs = torch.tensor(
      [
        self.cfg.rel_forward,
        self.cfg.rel_backward,
        self.cfg.rel_left,
        self.cfg.rel_right,
        self.cfg.rel_standing,
      ],
      device=self.device,
    )
    total = probs.sum()
    if total <= 0.0:
      msg = "DirectionalVelocityCommandCfg linear fractions sum to zero."
      raise ValueError(msg)
    cumsum = probs.cumsum(0)
    cat_idx = torch.searchsorted(cumsum, r * cumsum[-1])
    self._category[env_ids] = cat_idx

    # Reset command to zero and fill in the active linear axis.
    self.vel_command_b[env_ids] = 0.0
    r2 = torch.rand(n, device=self.device)

    for cat, (lo, hi), sign, dim in [
      (self.CAT_FWD,   self.cfg.forward_speed,  1, 0),
      (self.CAT_BWD,   self.cfg.backward_speed, -1, 0),
      (self.CAT_LEFT,  self.cfg.lateral_speed,   1, 1),
      (self.CAT_RIGHT, self.cfg.lateral_speed,  -1, 1),
    ]:
      mask = cat_idx == cat
      if mask.any():
        self.vel_command_b[env_ids[mask], dim] = sign * (
          lo + r2[mask] * (hi - lo)
        )

    # ── Angular overlay (independent probability, on fwd/bwd/standing) ───
    if self.cfg.rel_ang_overlay > 0 and self.cfg.ang_speed[1] > 0:
      r3 = torch.rand(n, device=self.device)
      fwd_bwd_stand = (
        (cat_idx == self.CAT_FWD)
        | (cat_idx == self.CAT_BWD)
        | (cat_idx == self.CAT_STAND)
      )
      ang_mask = fwd_bwd_stand & (r3 < self.cfg.rel_ang_overlay)
      if ang_mask.any():
        ang_ids = env_ids[ang_mask]
        m = int(ang_mask.sum())
        r4 = torch.rand(m, device=self.device)
        lo, hi = self.cfg.ang_speed
        sign = torch.where(
          torch.rand(m, device=self.device) < 0.5,
          torch.tensor(1.0, device=self.device),
          torch.tensor(-1.0, device=self.device),
        )
        self.vel_command_b[ang_ids, 2] = sign * (lo + r4 * (hi - lo))

  def _update_command(self) -> None:
    """No-op — the command is fixed between resets."""
    pass

  # ── GUI ────────────────────────────────────────────────────────────────

  def create_gui(
    self, name: str, server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
    on_change: Callable[..., None] | None = None,
    request_action: Callable[..., None] | None = None,
  ) -> None:
    from viser import Icon

    r = self.cfg
    axes = [
      ("lin_vel_x", float(r.forward_speed[1])),
      ("lin_vel_y", float(r.lateral_speed[1])),
      ("ang_vel_z",  float(r.ang_speed[1])),
    ]
    sliders: list = []
    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}", initial_value=max_val, step=0.1, min=0.0, max=10.0,
        )
        slider = server.gui.add_slider(
          label, min=-max_val, max=max_val, step=0.05, initial_value=0.0,
        )
        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value
        sliders.append(slider)
      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)
      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    """Step the command (no resampling — fixed per episode)."""
    self._update_metrics()
    self._update_command()
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value

  # ── Debug visualization ────────────────────────────────────────────────

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_mat_ws = matrix_from_quat(self.robot.data.root_link_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()
    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset
    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue
      def _l2w(v, p=base_pos_w, m=base_mat_w):
        return p + m @ v
      o = _l2w(np.array([0, 0, z_offset]) * scale)
      visualizer.add_arrow(
        o, _l2w((np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale),
        color=(0.2, 0.2, 0.6, 0.6), width=0.015,
      )
      visualizer.add_arrow(
        o, _l2w((np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale),
        color=(0.2, 0.6, 0.2, 0.6), width=0.015,
      )
      visualizer.add_arrow(
        o, _l2w((np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale),
        color=(0.0, 0.6, 1.0, 0.7), width=0.015,
      )
      visualizer.add_arrow(
        o, _l2w((np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale),
        color=(0.0, 1.0, 0.4, 0.7), width=0.015,
      )


@dataclass(kw_only=True)
class DirectionalVelocityCommandCfg(CommandTermCfg):
  """Per-reset directional velocity command with angular overlay.

  Each environment is randomly assigned (at reset) to one of five **linear**
  categories: forward (+x), backward (-x), left (+y), right (-y), or standing.
  Angular velocity ``ω_z`` is then **independently overlaid** onto forward,
  backward, and standing environments with probability ``rel_ang_overlay``
  (random ccw/cw sign), turning a straight-line walk into a curved one or
  letting the robot learn to turn in place.  The command
  ``[vx, vy, ω_z]`` is held constant until the next reset.

  The five linear fractions are normalised internally and should sum to ~1.0.
  """

  entity_name: str

  # Linear category fractions (normalised internally; sum to ~1.0).
  rel_forward: float = 0.25   # +x
  rel_backward: float = 0.25  # -x
  rel_left: float = 0.10      # +y
  rel_right: float = 0.10     # -y
  rel_standing: float = 0.30  # standing still

  # Angular overlay: independent probability that a forward, backward, or
  # standing env also receives an ``ω_z`` command (random ccw/cw sign).
  # Does not apply to lateral environments.
  rel_ang_overlay: float = 0.3

  # Speed ranges (absolute value, uniformly sampled).
  forward_speed: tuple[float, float] = (0.3, 3.0)
  backward_speed: tuple[float, float] = (0.1, 1.0)
  lateral_speed: tuple[float, float] = (0.1, 1.0)
  ang_speed: tuple[float, float] = (0.2, 1.0)

  # GUI / viz.
  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  """Unused — command is fixed per episode and never resampled on a timer."""

  def build(self, env: ManagerBasedRlEnv) -> DirectionalVelocityCommand:
    return DirectionalVelocityCommand(self, env)


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  rel_world_envs: float = 0.0
  """Fraction of environments that use world-frame velocity commands.
  World-frame envs sample linear velocity in world frame and rotate to body
  frame each step, so the command direction stays fixed in the world."""
  rel_forward_envs: float = 0.0
  """Fraction of environments that receive forward-only commands (positive
  lin_vel_x, zero lin_vel_y and ang_vel_z). Increases training coverage for
  straight-line walking, which is important for stair climbing."""
  init_velocity_prob: float = 0.0

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  limit_ranges: Ranges | None = None
  """Hard upper bounds for the adaptive ``velocity_cmd_levels`` curriculum.
  When ``None``, defaults to ``ranges`` (no expansion allowed)."""

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )


# ── Phased velocity command ────────────────────────────────────────────────


class PhasedVelocityCommand(CommandTerm):
  """Two-phase velocity command: directional → uniform.

  Phase 1 (iterations < ``phase_switch_iteration``):
      ``DirectionalVelocityCommand`` — per-reset directional categories
      (forward/backward/left/right/standing) with angular overlay.

  Phase 2 (iterations >= ``phase_switch_iteration``):
      ``UniformVelocityCommand`` — timer-based resampled uniform velocity
      ranges with heading/standing/world-frame modes.

  The phase gate reads ``env.common_step_counter`` which is persisted across
  checkpoints, so the transition survives training resume.

  Joystick (GUI) control is handled by the wrapper and applied to the active
  delegate's command tensor, so deployment behaviour is unchanged regardless
  of which phase is active.
  """

  cfg: "PhasedVelocityCommandCfg"

  def __init__(self, cfg: "PhasedVelocityCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    # Build both sub-commands as full CommandTerm instances.  They hold their
    # own independent state (vel_command_b, metrics, time_left, etc.) but are
    # never registered with the CommandManager — the wrapper mediates all
    # calls.
    self._directional: DirectionalVelocityCommand = cfg.directional_cfg.build(env)
    self._uniform: UniformVelocityCommand = cfg.uniform_cfg.build(env)

    # Track last-seen phase for one-shot transition logging.
    self._last_phase: int = 0  # 0 = directional, 1 = uniform

    # GUI state (owned by the wrapper; delegate GUIs are never created so
    # their joystick branches short-circuit).
    self._joystick_enabled: "viser.GuiCheckboxHandle | None" = None
    self._joystick_sliders: list["viser.GuiSliderHandle"] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  # ── Phase selection ────────────────────────────────────────────────────

  def _get_active(self) -> CommandTerm:
    """Return the delegate for the current training phase."""
    steps = self._env.common_step_counter
    threshold = self.cfg.phase_switch_iteration * self.cfg.steps_per_iteration
    new_phase = 1 if steps >= threshold else 0

    if new_phase != self._last_phase:
      phase_name = "Uniform" if new_phase == 1 else "Directional"
      print(
        f"[PhasedVelocityCommand] Phase -> {phase_name} at "
        f"env step {steps} (iteration ~{max(0, steps - 1) // self.cfg.steps_per_iteration})"
      )
      self._last_phase = new_phase

    return self._uniform if new_phase == 1 else self._directional

  # ── Public interface ───────────────────────────────────────────────────

  @property
  def command(self) -> torch.Tensor:
    """Body-frame velocity command ``[vx, vy, omega_z]`` from the active delegate."""
    return self._get_active().command

  @property
  def is_standing_env(self) -> torch.Tensor:
    """Boolean mask — which envs are standing (exposed for ``task_smp_product``)."""
    active = self._get_active()
    if isinstance(active, DirectionalVelocityCommand):
      return active.is_standing_env  # @property
    return active.is_standing_env  # tensor attribute on Uniform

  # ── Core lifecycle ─────────────────────────────────────────────────────

  def compute(self, dt: float) -> None:
    """Step the active delegate, then apply wrapper-level joystick override."""
    active = self._get_active()
    active.compute(dt)

    # Wrapper-owned joystick override (delegate GUIs are suppressed).
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      cmd = active.command  # live vel_command_b tensor
      for i, slider in enumerate(self._joystick_sliders):
        cmd[idx, i] = slider.value

  def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
    """Collect metrics from the active delegate, zero them, and resample."""
    active = self._get_active()
    extras: dict[str, float] = {}
    for metric_name, metric_value in active.metrics.items():
      extras[metric_name] = torch.mean(metric_value[env_ids]).item()
      metric_value[env_ids] = 0.0
    active.command_counter[env_ids] = 0
    active._resample(env_ids)
    return extras

  # ── Abstract method stubs (never called; delegates handle their own) ───

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    pass

  def _update_command(self) -> None:
    pass

  # ── Debug visualization ────────────────────────────────────────────────

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    self._get_active()._debug_vis_impl(visualizer)

  # ── GUI ────────────────────────────────────────────────────────────────

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
    on_change: Callable[..., None] | None = None,
    request_action: Callable[..., None] | None = None,
  ) -> None:
    """Wrapper-owned joystick sliders (delegates' own GUIs are never created).

    The default max values cover both phases' speed ranges; the user can
    expand them via the per-axis Max sliders.
    """
    from viser import Icon

    axes = [
      ("lin_vel_x", 5.0),
      ("lin_vel_y", 3.0),
      ("ang_vel_z", 2.0),
    ]
    sliders: list = []
    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}", initial_value=max_val, step=0.1, min=0.0, max=10.0,
        )
        slider = server.gui.add_slider(
          label, min=-max_val, max=max_val, step=0.05, initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx


@dataclass(kw_only=True)
class PhasedVelocityCommandCfg(CommandTermCfg):
  """Configuration for ``PhasedVelocityCommand``.

  Wraps a ``DirectionalVelocityCommand`` (used for the first
  ``phase_switch_iteration`` training iterations) and a
  ``UniformVelocityCommand`` (used thereafter).  The switch is gated on the
  environment-global ``common_step_counter``, so it survives checkpoint
  save/load automatically.
  """

  directional_cfg: DirectionalVelocityCommandCfg
  """Phase 1 command — fixed per episode, directional categories."""

  uniform_cfg: UniformVelocityCommandCfg
  """Phase 2 command — timer-based resampled uniform distribution."""

  phase_switch_iteration: int
  """Training iteration index *after which* we switch to uniform.

  Directional is used for iterations ``[0, phase_switch_iteration)`` and
  uniform for ``[phase_switch_iteration, max_iterations]``.  Set to ``0`` to
  use uniform from the start.
  """

  steps_per_iteration: int = 24
  """Env steps per training iteration.  Must match
  ``RslRlOnPolicyRunnerCfg.num_steps_per_env`` in the RL config."""

  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  """Unused — the active delegate manages its own resampling timer."""

  def build(self, env: ManagerBasedRlEnv) -> PhasedVelocityCommand:
    return PhasedVelocityCommand(self, env)
