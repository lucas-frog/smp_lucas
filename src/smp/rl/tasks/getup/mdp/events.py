"""Reset events for the getup task."""

from __future__ import annotations

import math

import torch
from mjlab.envs import ManagerBasedRlEnv

from smp.rl.events import gsi_reset

__all__ = [
  "mixed_gsi_fallen_pose_reset",
  "reset_real_fallen_poses",
  "reset_stand_counter",
]


def _quat_from_euler_xyz(
  roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
) -> torch.Tensor:
  """Return ``wxyz`` quaternions for XYZ Euler angles."""
  half_roll = 0.5 * roll
  half_pitch = 0.5 * pitch
  half_yaw = 0.5 * yaw

  cr, sr = torch.cos(half_roll), torch.sin(half_roll)
  cp, sp = torch.cos(half_pitch), torch.sin(half_pitch)
  cy, sy = torch.cos(half_yaw), torch.sin(half_yaw)

  return torch.stack(
    (
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
    ),
    dim=-1,
  )


def _default_joint_pos(robot, env_ids: torch.Tensor) -> torch.Tensor:
  data = robot.data
  default_joint_pos = data.default_joint_pos
  if default_joint_pos.ndim == 1:
    return default_joint_pos.unsqueeze(0).expand(env_ids.numel(), -1).clone()
  return default_joint_pos[env_ids].clone()


def _write_robot_state(robot, root_state, joint_pos, joint_vel, env_ids) -> None:
  if hasattr(robot, "write_root_state_to_sim"):
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
  else:
    robot.data.write_root_state(root_state, env_ids=env_ids)

  if hasattr(robot, "write_joint_state_to_sim"):
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
  else:
    robot.data.write_joint_state(joint_pos, joint_vel, env_ids=env_ids)


def _clamp_to_joint_limits(robot, joint_pos: torch.Tensor, env_ids: torch.Tensor):
  limits = getattr(robot.data, "soft_joint_pos_limits", None)
  if limits is None:
    limits = getattr(robot.data, "joint_pos_limits", None)
  if limits is None:
    return joint_pos
  if limits.ndim == 3:
    limits = limits[env_ids]
  lo = limits[..., 0] + 1e-3
  hi = limits[..., 1] - 1e-3
  return torch.minimum(torch.maximum(joint_pos, lo), hi)


def _reset_smp_buffer_from_fallen_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  root_state: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
) -> None:
  if not hasattr(env, "_smp_buffer"):
    return

  if hasattr(env, "sim") and hasattr(env.sim, "forward"):
    env.sim.forward()

  robot = env.scene["robot"]
  origins = env.scene.env_origins[env_ids]
  buffer = env._smp_buffer  # type: ignore[attr-defined]
  window_size = buffer.window_size

  root_pos = root_state[:, 0:3] - origins
  root_quat = root_state[:, 3:7]
  lin_vel = root_state[:, 7:10]
  ang_vel = root_state[:, 10:13]

  ee_indexes = getattr(env, "_smp_ee_indexes", None)
  body_pos = getattr(robot.data, "body_link_pos_w", None)
  if ee_indexes is not None and body_pos is not None:
    ee_pos = body_pos[env_ids][:, ee_indexes] - origins[:, None, :]
  else:
    ee_pos = root_pos[:, None, :].expand(-1, buffer.num_ee, -1).clone()

  buffer.reset(
    env_ids,
    root_pos[:, None, :].expand(-1, window_size, -1).clone(),
    root_quat[:, None, :].expand(-1, window_size, -1).clone(),
    lin_vel[:, None, :].expand(-1, window_size, -1).clone(),
    ang_vel[:, None, :].expand(-1, window_size, -1).clone(),
    ee_pos[:, None, :, :].expand(-1, window_size, -1, -1).clone(),
    joint_pos[:, None, :].expand(-1, window_size, -1).clone(),
    joint_vel[:, None, :].expand(-1, window_size, -1).clone(),
  )


@torch.no_grad()
def reset_real_fallen_poses(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  root_height_range: tuple[float, float] = (0.22, 0.34),
  xy_jitter: float = 0.15,
  tilt_noise: float = 0.12,
  joint_noise: float = 0.18,
  velocity_noise: float = 0.05,
) -> None:
  """Reset G1 into low, real-world fallen poses instead of GSI samples.

  The reset distribution covers supine, prone, left-side, and right-side starts
  with small yaw, tilt, joint, and velocity noise.  It also re-primes the SMP
  feature buffer from the written fallen state so the first reward steps do not
  mix the new pose with stale GSI windows.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = int(env_ids.numel())
  if n == 0:
    return

  device = torch.device(env.device)
  robot = env.scene["robot"]
  origins = env.scene.env_origins[env_ids]

  pose_ids = torch.randint(0, 4, (n,), device=device)
  base_roll = torch.zeros(n, device=device)
  base_pitch = torch.zeros(n, device=device)
  base_pitch[pose_ids == 0] = math.pi / 2.0
  base_pitch[pose_ids == 1] = -math.pi / 2.0
  base_roll[pose_ids == 2] = math.pi / 2.0
  base_roll[pose_ids == 3] = -math.pi / 2.0

  roll = base_roll + (torch.rand(n, device=device) * 2.0 - 1.0) * tilt_noise
  pitch = base_pitch + (torch.rand(n, device=device) * 2.0 - 1.0) * tilt_noise
  yaw = (torch.rand(n, device=device) * 2.0 - 1.0) * math.pi
  quat = _quat_from_euler_xyz(roll, pitch, yaw)

  root_state = torch.zeros(n, 13, device=device)
  root_state[:, 0:3] = origins
  root_state[:, 0:2] += (torch.rand(n, 2, device=device) * 2.0 - 1.0) * xy_jitter
  root_state[:, 2] = torch.empty(n, device=device).uniform_(*root_height_range)
  root_state[:, 3:7] = quat
  root_state[:, 7:13] = torch.randn(n, 6, device=device) * velocity_noise

  joint_pos = _default_joint_pos(robot, env_ids)
  if joint_pos.shape[1] >= 29:
    crouched_offsets = torch.zeros_like(joint_pos)
    crouched_offsets[:, [0, 6]] = -0.45
    crouched_offsets[:, [3, 9]] = 0.65
    crouched_offsets[:, [4, 10]] = -0.25
    crouched_offsets[:, [15, 22]] = 0.35
    crouched_offsets[:, [16, 23]] = torch.tensor(
      [0.25, -0.25], device=device
    )
    crouched_offsets[:, [18, 25]] = 0.35
    joint_pos += crouched_offsets
  joint_pos += torch.randn_like(joint_pos) * joint_noise
  joint_pos = _clamp_to_joint_limits(robot, joint_pos, env_ids)
  joint_vel = torch.randn_like(joint_pos) * velocity_noise

  _write_robot_state(robot, root_state, joint_pos, joint_vel, env_ids)
  _reset_smp_buffer_from_fallen_state(env, env_ids, root_state, joint_pos, joint_vel)


def _apply_mixed_reset_once(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  *,
  gsi_ratio: float,
  fallen_pose_params: dict[str, object] | None,
) -> None:
  n = int(env_ids.numel())
  if n == 0:
    return

  num_gsi = int(math.floor(n * gsi_ratio + 0.5))
  num_gsi = min(max(num_gsi, 0), n)
  gsi_env_ids = env_ids[:num_gsi]
  fallen_env_ids = env_ids[num_gsi:]

  if gsi_env_ids.numel() > 0:
    gsi_reset(env, gsi_env_ids)
  if fallen_env_ids.numel() > 0:
    reset_real_fallen_poses(env, fallen_env_ids, **(fallen_pose_params or {}))


@torch.no_grad()
def mixed_gsi_fallen_pose_reset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  gsi_ratio: float = 0.5,
  fallen_pose_params: dict[str, object] | None = None,
) -> None:
  """Split reset envs between GSI samples and real fallen poses."""
  if not 0.0 <= gsi_ratio <= 1.0:
    msg = f"gsi_ratio must be in [0, 1], got {gsi_ratio}."
    raise ValueError(msg)

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = int(env_ids.numel())
  if n == 0:
    return

  _apply_mixed_reset_once(
    env,
    env_ids,
    gsi_ratio=gsi_ratio,
    fallen_pose_params=fallen_pose_params,
  )


# 在大规模并行的强化学习仿真环境中，将指定机器人的“持续站立时间计数器”清零
@torch.no_grad()
def reset_stand_counter(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None
) -> None:
  """Zero the ``stood_up`` standing-hold counter for the reset envs (no-op until
  ``stood_up`` lazily creates it).  Separate from ``gsi_reset`` so it stays reusable."""
  if not hasattr(env, "_getup_stand_count"):
    return
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env._getup_stand_count[env_ids] = 0  # type: ignore[attr-defined]
