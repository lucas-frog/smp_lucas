"""Recovery dual-task command and reward helpers."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  subtract_frame_transforms,
  yaw_quat,
)

from smp.rl.rewards import task_smp_product

if TYPE_CHECKING:
  from collections.abc import Callable

  from mjlab.envs import ManagerBasedRlEnv

  TaskTerm = tuple[Callable[..., torch.Tensor], float, dict]


TRAJECTORY_TASK_ID = 0.0
VELOCITY_TASK_ID = 1.0
_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))

TRACKING_EXP_COMPONENTS = (
  "position",
  "rotation",
  "linear_velocity",
  "angular_velocity",
  "root_height",
)
TRACKING_COMPONENTS = (*TRACKING_EXP_COMPONENTS, "energy")
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def robot_root_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Root height above the ground plane."""
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


def _body_ids(asset_cfg: SceneEntityCfg) -> list[int] | slice:
  return asset_cfg.body_ids


def _root_quat_for_bodies(root_quat_w: torch.Tensor, body_count: int) -> torch.Tensor:
  return root_quat_w[:, None, :].expand(-1, body_count, -1)


def recovery_body_pos_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Selected body positions in the root local frame."""
  asset = env.scene[asset_cfg.name]
  body_pos_w = asset.data.body_link_pos_w[:, _body_ids(asset_cfg)]
  root_pos_w = asset.data.root_link_pos_w[:, None, :]
  root_quat_w = _root_quat_for_bodies(asset.data.root_link_quat_w, body_pos_w.shape[1])
  body_pos_b = quat_apply_inverse(root_quat_w, body_pos_w - root_pos_w)
  return body_pos_b.reshape(body_pos_b.shape[0], -1)


def recovery_body_rot_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Selected body rotations in the root local frame as 6D matrix columns."""
  asset = env.scene[asset_cfg.name]
  body_quat_w = asset.data.body_link_quat_w[:, _body_ids(asset_cfg)]
  root_quat_inv = quat_conjugate(
    _root_quat_for_bodies(asset.data.root_link_quat_w, body_quat_w.shape[1])
  )
  body_quat_b = quat_mul(root_quat_inv, body_quat_w)
  body_rot_b = matrix_from_quat(body_quat_b)
  return body_rot_b[..., :2].reshape(body_rot_b.shape[0], -1)


def recovery_body_lin_vel_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Selected body linear velocities expressed in the root local frame."""
  asset = env.scene[asset_cfg.name]
  body_lin_vel_w = asset.data.body_link_lin_vel_w[:, _body_ids(asset_cfg)]
  root_quat_w = _root_quat_for_bodies(
    asset.data.root_link_quat_w, body_lin_vel_w.shape[1]
  )
  body_lin_vel_b = quat_apply_inverse(root_quat_w, body_lin_vel_w)
  return body_lin_vel_b.reshape(body_lin_vel_b.shape[0], -1)


def recovery_body_ang_vel_b(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Selected body angular velocities expressed in the root local frame."""
  asset = env.scene[asset_cfg.name]
  body_ang_vel_w = asset.data.body_link_ang_vel_w[:, _body_ids(asset_cfg)]
  root_quat_w = _root_quat_for_bodies(
    asset.data.root_link_quat_w, body_ang_vel_w.shape[1]
  )
  body_ang_vel_b = quat_apply_inverse(root_quat_w, body_ang_vel_w)
  return body_ang_vel_b.reshape(body_ang_vel_b.shape[0], -1)


@dataclass(frozen=True)
class RecoveryTaskTensors:
  """Static per-env recovery task assignment tensors."""

  is_trajectory_env: torch.Tensor
  is_velocity_env: torch.Tensor
  task_id: torch.Tensor


def build_recovery_task_tensors(
  *,
  num_envs: int,
  trajectory_fraction: float,
  device: torch.device | str,
  trajectory_task_id: float = TRAJECTORY_TASK_ID,
  velocity_task_id: float = VELOCITY_TASK_ID,
) -> RecoveryTaskTensors:
  """Build deterministic recovery task masks and critic task ids."""
  if not 0.0 <= trajectory_fraction <= 1.0:
    msg = f"trajectory_fraction must be in [0, 1], got {trajectory_fraction}."
    raise ValueError(msg)

  device = torch.device(device)
  num_trajectory = int(round(num_envs * trajectory_fraction))
  is_trajectory_env = torch.zeros(num_envs, dtype=torch.bool, device=device)
  is_trajectory_env[:num_trajectory] = True
  is_velocity_env = ~is_trajectory_env

  task_id = torch.full((num_envs, 1), velocity_task_id, dtype=torch.float, device=device)
  task_id[is_trajectory_env, 0] = trajectory_task_id
  return RecoveryTaskTensors(
    is_trajectory_env=is_trajectory_env,
    is_velocity_env=is_velocity_env,
    task_id=task_id,
  )


def compose_recovery_command(
  *,
  velocity_command: torch.Tensor,
  trajectory_velocity_command: torch.Tensor,
  trajectory_future_velocity_command: torch.Tensor | None = None,
  trajectory_is_standing: torch.Tensor | None = None,
  trajectory_command: torch.Tensor | None,
  is_velocity_env: torch.Tensor,
  is_trajectory_env: torch.Tensor,
  trajectory_future_velocity_min_speed: float = 0.1,
) -> torch.Tensor:
  """Compose the shared recovery command observation.

  The first slots keep the velocity command layout so existing velocity reward
  terms can read ``[vx, vy, omega_z]``. Tracking environments use the reference
  root velocity once the reference has started walking. During get-up frames,
  they use the post-standing reference velocity, falling back to the sampled
  velocity command when the reference never walks.
  """
  del is_velocity_env, trajectory_command

  # Old tracking command layout kept for rollback:
  # command = velocity_command.new_zeros(
  #   velocity_command.shape[0],
  #   velocity_command.shape[1] + trajectory_command.shape[1],
  # )
  # command[is_velocity_env, : velocity_command.shape[1]] = velocity_command[
  #   is_velocity_env
  # ]
  # command[
  #   is_trajectory_env, : trajectory_velocity_command.shape[1]
  # ] = trajectory_velocity_command[is_trajectory_env].to(command.dtype)
  # command[is_trajectory_env, velocity_command.shape[1] :] = trajectory_command[
  #   is_trajectory_env
  # ].to(command.dtype)
  # return command

  command = velocity_command.clone()
  trajectory_command_velocity = select_trajectory_recovery_velocity_command(
    trajectory_velocity_command=trajectory_velocity_command,
    trajectory_future_velocity_command=trajectory_future_velocity_command,
    fallback_velocity_command=velocity_command,
    trajectory_is_standing=trajectory_is_standing,
    min_speed=trajectory_future_velocity_min_speed,
  )
  command[is_trajectory_env] = 0.0
  command[
    is_trajectory_env, : trajectory_command_velocity.shape[1]
  ] = trajectory_command_velocity[is_trajectory_env].to(command.dtype)
  return command


def command_speed(command: torch.Tensor) -> torch.Tensor:
  """Return scalar xy+yaw command magnitude."""
  return torch.linalg.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])


def select_trajectory_recovery_velocity_command(
  *,
  trajectory_velocity_command: torch.Tensor,
  trajectory_future_velocity_command: torch.Tensor | None,
  fallback_velocity_command: torch.Tensor,
  trajectory_is_standing: torch.Tensor | None,
  min_speed: float,
) -> torch.Tensor:
  """Select the command shown to trajectory environments.

  Standing/walking frames keep the current reference root velocity when it is
  nonzero. Get-up frames receive the future post-standing velocity. If the
  reference never moves after standing, the sampled velocity command becomes the
  intended post-recovery command.
  """
  if trajectory_future_velocity_command is None or trajectory_is_standing is None:
    return trajectory_velocity_command

  future_speed = command_speed(trajectory_future_velocity_command)
  current_speed = command_speed(trajectory_velocity_command)
  valid_future = future_speed > min_speed
  valid_current = current_speed > min_speed

  future_or_fallback = torch.where(
    valid_future.unsqueeze(-1),
    trajectory_future_velocity_command,
    fallback_velocity_command,
  )
  standing_command = torch.where(
    valid_current.unsqueeze(-1),
    trajectory_velocity_command,
    future_or_fallback,
  )
  return torch.where(
    trajectory_is_standing.unsqueeze(-1),
    standing_command,
    future_or_fallback,
  )


def trajectory_velocity_command_from_tensors(
  root_quat_w: torch.Tensor,
  root_lin_vel_w: torch.Tensor,
  root_ang_vel_w: torch.Tensor,
) -> torch.Tensor:
  """Return body-frame ``[vx, vy, omega_z]`` from reference root velocities."""
  root_lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
  root_ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
  return torch.stack(
    [root_lin_vel_b[:, 0], root_lin_vel_b[:, 1], root_ang_vel_b[:, 2]],
    dim=-1,
  )


def root_pose_goal_command(
  *,
  current_root_pos_w: torch.Tensor,
  current_root_quat_w: torch.Tensor,
  target_root_pos_w: torch.Tensor,
  target_root_quat_w: torch.Tensor,
) -> torch.Tensor:
  """Encode a target root pose in the current root local frame.

  Layout: ``[pos_b.xyz, rot_b.6d]`` where the rotation uses the first two
  columns of the relative rotation matrix, matching existing orientation
  observations.
  """
  target_pos_b, target_quat_b = subtract_frame_transforms(
    current_root_pos_w,
    current_root_quat_w,
    target_root_pos_w,
    target_root_quat_w,
  )
  target_rot_b = matrix_from_quat(target_quat_b)
  target_ori_6d = target_rot_b[..., :2].reshape(target_rot_b.shape[0], -1)
  return torch.cat([target_pos_b, target_ori_6d], dim=-1)


def tracking_future_root_goal_command(
  *,
  robot_root_pos_w: torch.Tensor,
  robot_root_quat_w: torch.Tensor,
  ref_root_pos_w: torch.Tensor,
  ref_root_quat_w: torch.Tensor,
  ref_future_root_pos_w: torch.Tensor,
  ref_future_root_quat_w: torch.Tensor,
) -> torch.Tensor:
  """Encode a future reference root pose in the robot root local frame.

  The future reference is yaw-aligned to the current robot root, mirroring the
  tracking reference alignment used by the full-body rewards.
  """
  delta_pos_w = robot_root_pos_w.clone()
  delta_pos_w[:, 2] = ref_root_pos_w[:, 2]
  delta_ori_w = yaw_quat(quat_mul(robot_root_quat_w, quat_inv(ref_root_quat_w)))
  target_root_pos_w = delta_pos_w + quat_apply(
    delta_ori_w,
    ref_future_root_pos_w - ref_root_pos_w,
  )
  target_root_quat_w = quat_mul(delta_ori_w, ref_future_root_quat_w)
  return root_pose_goal_command(
    current_root_pos_w=robot_root_pos_w,
    current_root_quat_w=robot_root_quat_w,
    target_root_pos_w=target_root_pos_w,
    target_root_quat_w=target_root_quat_w,
  )


def velocity_goal_command(
  *,
  velocity_command: torch.Tensor,
  current_root_pos_w: torch.Tensor,
  current_root_quat_w: torch.Tensor,
  root_height: torch.Tensor,
  horizon_s: float,
  standing_height: float,
  stand_up_threshold: float,
  stand_up_speed: torch.Tensor | float,
) -> torch.Tensor:
  """Convert a sampled local twist into a future local root-pose goal.

  Upright SMP envs receive the 1s pose implied by the sampled velocity. Low
  envs first receive a stand-up root goal at ``standing_height`` with yaw-only
  upright orientation.
  """
  del current_root_pos_w
  device = velocity_command.device
  dtype = velocity_command.dtype
  horizon = torch.as_tensor(horizon_s, device=device, dtype=dtype)
  vx = velocity_command[:, 0]
  vy = velocity_command[:, 1]
  wz = velocity_command[:, 2]
  yaw = wz * horizon
  sin_yaw = torch.sin(yaw)
  cos_yaw = torch.cos(yaw)
  nonzero_turn = torch.abs(wz) > 1.0e-6
  safe_wz = torch.where(nonzero_turn, wz, torch.ones_like(wz))

  dx_turn = (vx * sin_yaw + vy * (cos_yaw - 1.0)) / safe_wz
  dy_turn = (vx * (1.0 - cos_yaw) + vy * sin_yaw) / safe_wz
  dx = torch.where(nonzero_turn, dx_turn, vx * horizon)
  dy = torch.where(nonzero_turn, dy_turn, vy * horizon)

  goal_pos_b = torch.stack([dx, dy, torch.zeros_like(dx)], dim=-1)
  goal_quat_b = quat_from_euler_xyz(
    torch.zeros_like(yaw),
    torch.zeros_like(yaw),
    yaw,
  )

  stand_pos_b = torch.zeros_like(goal_pos_b)
  height_delta = torch.clamp(standing_height - root_height.reshape(-1), min=0.0)
  stand_up_speed = torch.as_tensor(stand_up_speed, device=device, dtype=dtype)
  stand_pos_b[:, 2] = torch.clamp(height_delta, max=stand_up_speed * horizon)
  stand_quat_b = quat_mul(
    quat_inv(current_root_quat_w),
    yaw_quat(current_root_quat_w),
  )

  low = root_height.reshape(-1) < stand_up_threshold
  goal_pos_b = torch.where(low.unsqueeze(-1), stand_pos_b, goal_pos_b)
  goal_quat_b = torch.where(low.unsqueeze(-1), stand_quat_b, goal_quat_b)

  goal_rot_b = matrix_from_quat(goal_quat_b)
  goal_ori_6d = goal_rot_b[..., :2].reshape(goal_rot_b.shape[0], -1)
  return torch.cat([goal_pos_b, goal_ori_6d], dim=-1)


def compose_recovery_goal_command(
  *,
  trajectory_goal_command: torch.Tensor,
  velocity_goal_command: torch.Tensor,
  is_trajectory_env: torch.Tensor,
) -> torch.Tensor:
  """Blend tracking and SMP goal-condition commands into one observation."""
  command = velocity_goal_command.clone()
  command[is_trajectory_env] = trajectory_goal_command[is_trajectory_env].to(
    command.dtype
  )
  return command


def future_standing_velocity_commands_from_tensors(
  *,
  root_height: torch.Tensor,
  velocity_command: torch.Tensor,
  standing_height: float,
  window_steps: int,
) -> torch.Tensor:
  """Average the first future standing velocity window for every motion frame."""
  root_height = root_height.reshape(-1)
  window_steps = max(1, int(window_steps))
  future_command = torch.zeros_like(velocity_command)
  standing_steps = (root_height >= standing_height).nonzero(as_tuple=False).flatten()
  if standing_steps.numel() == 0:
    return future_command

  num_steps = velocity_command.shape[0]
  for step in range(num_steps):
    future_steps = standing_steps[standing_steps >= step]
    if future_steps.numel() == 0:
      continue
    start = int(future_steps[0].item())
    end = min(start + window_steps, num_steps)
    future_command[step] = velocity_command[start:end].mean(dim=0)
  return future_command


def _reference_delta_yaw(
  ref_anchor_quat_w: torch.Tensor,
  robot_anchor_quat_w: torch.Tensor,
) -> torch.Tensor:
  return yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat_w)))


def align_reference_body_state(
  *,
  ref_body_pos_w: torch.Tensor,
  ref_body_quat_w: torch.Tensor,
  ref_anchor_pos_w: torch.Tensor,
  ref_anchor_quat_w: torch.Tensor,
  robot_anchor_pos_w: torch.Tensor,
  robot_anchor_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Align reference bodies to the robot anchor position and yaw.

  This mirrors the tracking task: the reference motion keeps its anchor height,
  but its xy position and yaw are shifted to the current robot anchor. That keeps
  imitation relative to the robot instead of demanding an absolute world pose.
  """
  num_bodies = ref_body_pos_w.shape[1]
  ref_anchor_pos = ref_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
  ref_anchor_quat = ref_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
  robot_anchor_pos = robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
  robot_anchor_quat = robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)

  delta_pos_w = robot_anchor_pos.clone()
  delta_pos_w[..., 2] = ref_anchor_pos[..., 2]
  delta_ori_w = _reference_delta_yaw(ref_anchor_quat, robot_anchor_quat)

  body_quat_relative_w = quat_mul(delta_ori_w, ref_body_quat_w)
  body_pos_relative_w = delta_pos_w + quat_apply(
    delta_ori_w, ref_body_pos_w - ref_anchor_pos
  )
  return body_pos_relative_w, body_quat_relative_w


def _trajectory_task_command(env: ManagerBasedRlEnv, command_name: str):
  command = env.command_manager.get_term(command_name)
  assert hasattr(command, "is_trajectory_env")
  return command


def _mask_trajectory_envs(command, value: torch.Tensor) -> torch.Tensor:
  mask = command.is_trajectory_env
  while mask.ndim < value.ndim:
    mask = mask.unsqueeze(-1)
  return torch.where(mask, value, torch.zeros_like(value))


def tracking_only_action_rate_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Penalize action changes only for recovery trajectory environments."""
  command = _trajectory_task_command(env, command_name)
  reward = torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )
  return _mask_trajectory_envs(command, reward)


def tracking_only_joint_pos_limits(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize soft joint-limit violations only on trajectory environments."""
  command = _trajectory_task_command(env, command_name)
  asset = env.scene[asset_cfg.name]
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None
  out_of_limits = -(
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
  ).clip(max=0.0)
  out_of_limits += (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
  ).clip(min=0.0)
  return _mask_trajectory_envs(command, torch.sum(out_of_limits, dim=1))


def tracking_only_self_collision_cost(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions only for recovery trajectory environments."""
  command = _trajectory_task_command(env, command_name)
  reward = self_collision_cost(
    env,
    sensor_name=sensor_name,
    force_threshold=force_threshold,
  )
  return _mask_trajectory_envs(command, reward)


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor pose expressed in the current robot anchor frame."""
  command = _trajectory_task_command(env, command_name)
  traj = command.trajectory
  pos, _ = subtract_frame_transforms(
    traj.robot_anchor_pos_w,
    traj.robot_anchor_quat_w,
    traj.anchor_pos_w,
    traj.anchor_quat_w,
  )
  return _mask_trajectory_envs(command, pos.view(env.num_envs, -1))


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor orientation in the robot anchor frame as 6D rotation."""
  command = _trajectory_task_command(env, command_name)
  traj = command.trajectory
  _, ori = subtract_frame_transforms(
    traj.robot_anchor_pos_w,
    traj.robot_anchor_quat_w,
    traj.anchor_pos_w,
    traj.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return _mask_trajectory_envs(command, mat[..., :2].reshape(mat.shape[0], -1))


def robot_body_pos_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Tracked robot body positions in the robot anchor frame."""
  del asset_cfg
  command = _trajectory_task_command(env, command_name)
  traj = command.trajectory
  num_bodies = len(traj.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    traj.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    traj.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    traj.robot_body_pos_w,
    traj.robot_body_quat_w,
  )
  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Tracked robot body orientations in the robot anchor frame as 6D rotation."""
  del asset_cfg
  command = _trajectory_task_command(env, command_name)
  traj = command.trajectory
  num_bodies = len(traj.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    traj.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    traj.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    traj.robot_body_pos_w,
    traj.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_lin_vel_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Tracked robot body linear velocities in the robot anchor frame."""
  del asset_cfg
  command = _trajectory_task_command(env, command_name)
  traj = command.trajectory
  num_bodies = len(traj.cfg.body_names)
  anchor_quat_w = traj.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
  lin_vel_b = quat_apply_inverse(anchor_quat_w, traj.robot_body_lin_vel_w)
  return lin_vel_b.view(env.num_envs, -1)


def robot_body_ang_vel_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Tracked robot body angular velocities in the robot anchor frame."""
  del asset_cfg
  command = _trajectory_task_command(env, command_name)
  traj = command.trajectory
  num_bodies = len(traj.cfg.body_names)
  anchor_quat_w = traj.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
  ang_vel_b = quat_apply_inverse(anchor_quat_w, traj.robot_body_ang_vel_w)
  return ang_vel_b.view(env.num_envs, -1)


def _tracking_body_indexes(
  trajectory_command,
  body_names: tuple[str, ...] | None,
) -> list[int]:
  return [
    i
    for i, name in enumerate(trajectory_command.cfg.body_names)
    if body_names is None or name in body_names
  ]


def _identity_quat_like(quat: torch.Tensor) -> torch.Tensor:
  identity = torch.zeros_like(quat)
  identity[..., 0] = 1.0
  return identity


def _repeat_anchor_frame(
  anchor_pos_w: torch.Tensor,
  anchor_quat_w: torch.Tensor,
  body_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  return (
    anchor_pos_w[:, None, :].repeat(1, body_count, 1),
    anchor_quat_w[:, None, :].repeat(1, body_count, 1),
  )


def tracking_anchor_pose_b(trajectory_command) -> tuple[torch.Tensor, ...]:
  """Reference and robot anchor poses expressed in the robot anchor frame."""
  ref_anchor_pos_b, ref_anchor_quat_b = subtract_frame_transforms(
    trajectory_command.robot_anchor_pos_w,
    trajectory_command.robot_anchor_quat_w,
    trajectory_command.anchor_pos_w,
    trajectory_command.anchor_quat_w,
  )
  robot_anchor_pos_b = torch.zeros_like(ref_anchor_pos_b)
  robot_anchor_quat_b = _identity_quat_like(ref_anchor_quat_b)
  return ref_anchor_pos_b, ref_anchor_quat_b, robot_anchor_pos_b, robot_anchor_quat_b


def tracking_body_pose_b(trajectory_command) -> tuple[torch.Tensor, ...]:
  """Reference and robot body poses expressed in the robot anchor frame."""
  num_bodies = len(trajectory_command.cfg.body_names)
  anchor_pos_w, anchor_quat_w = _repeat_anchor_frame(
    trajectory_command.robot_anchor_pos_w,
    trajectory_command.robot_anchor_quat_w,
    num_bodies,
  )
  ref_body_pos_b, ref_body_quat_b = subtract_frame_transforms(
    anchor_pos_w,
    anchor_quat_w,
    trajectory_command.body_pos_relative_w,
    trajectory_command.body_quat_relative_w,
  )
  robot_body_pos_b, robot_body_quat_b = subtract_frame_transforms(
    anchor_pos_w,
    anchor_quat_w,
    trajectory_command.robot_body_pos_w,
    trajectory_command.robot_body_quat_w,
  )
  return ref_body_pos_b, ref_body_quat_b, robot_body_pos_b, robot_body_quat_b


def tracking_body_velocity_b(trajectory_command) -> tuple[torch.Tensor, ...]:
  """Reference and robot body velocities expressed in the robot anchor frame."""
  num_bodies = len(trajectory_command.cfg.body_names)
  _, anchor_quat_w = _repeat_anchor_frame(
    trajectory_command.robot_anchor_pos_w,
    trajectory_command.robot_anchor_quat_w,
    num_bodies,
  )
  delta_yaw_w = _reference_delta_yaw(
    trajectory_command.anchor_quat_w,
    trajectory_command.robot_anchor_quat_w,
  )[:, None, :].repeat(1, num_bodies, 1)
  ref_body_lin_vel_b = quat_apply_inverse(
    anchor_quat_w, quat_apply(delta_yaw_w, trajectory_command.body_lin_vel_w)
  )
  robot_body_lin_vel_b = quat_apply_inverse(
    anchor_quat_w, trajectory_command.robot_body_lin_vel_w
  )
  ref_body_ang_vel_b = quat_apply_inverse(
    anchor_quat_w, quat_apply(delta_yaw_w, trajectory_command.body_ang_vel_w)
  )
  robot_body_ang_vel_b = quat_apply_inverse(
    anchor_quat_w, trajectory_command.robot_body_ang_vel_w
  )
  return (
    ref_body_lin_vel_b,
    robot_body_lin_vel_b,
    ref_body_ang_vel_b,
    robot_body_ang_vel_b,
  )


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _trajectory_task_command(env, command_name)
  traj = command.trajectory
  ref_anchor_pos_b, _, robot_anchor_pos_b, _ = tracking_anchor_pose_b(traj)
  error = torch.sum(torch.square(ref_anchor_pos_b - robot_anchor_pos_b), dim=-1)
  return _mask_trajectory_envs(command, torch.exp(-error / std**2))


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _trajectory_task_command(env, command_name)
  traj = command.trajectory
  _, ref_anchor_quat_b, _, robot_anchor_quat_b = tracking_anchor_pose_b(traj)
  error = quat_error_magnitude(ref_anchor_quat_b, robot_anchor_quat_b) ** 2
  return _mask_trajectory_envs(command, torch.exp(-error / std**2))


def motion_root_height_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  root_body_index: int = 0,
) -> torch.Tensor:
  command = _trajectory_task_command(env, command_name)
  traj = command.trajectory
  error = torch.square(
    traj.body_pos_w[:, root_body_index, 2]
    - traj.robot_body_pos_w[:, root_body_index, 2]
  )
  return _mask_trajectory_envs(command, torch.exp(-error / std**2))


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = _trajectory_task_command(env, command_name)
  traj = command.trajectory
  body_indexes = _tracking_body_indexes(traj, body_names)
  ref_body_pos_b, _, robot_body_pos_b, _ = tracking_body_pose_b(traj)
  error = torch.sum(
    torch.square(
      ref_body_pos_b[:, body_indexes]
      - robot_body_pos_b[:, body_indexes]
    ),
    dim=-1,
  )
  return _mask_trajectory_envs(command, torch.exp(-error.mean(-1) / std**2))


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = _trajectory_task_command(env, command_name)
  traj = command.trajectory
  body_indexes = _tracking_body_indexes(traj, body_names)
  _, ref_body_quat_b, _, robot_body_quat_b = tracking_body_pose_b(traj)
  error = (
    quat_error_magnitude(
      ref_body_quat_b[:, body_indexes],
      robot_body_quat_b[:, body_indexes],
    )
    ** 2
  )
  return _mask_trajectory_envs(command, torch.exp(-error.mean(-1) / std**2))


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = _trajectory_task_command(env, command_name)
  traj = command.trajectory
  body_indexes = _tracking_body_indexes(traj, body_names)
  ref_body_lin_vel_b, robot_body_lin_vel_b, _, _ = tracking_body_velocity_b(traj)
  error = torch.sum(
    torch.square(
      ref_body_lin_vel_b[:, body_indexes]
      - robot_body_lin_vel_b[:, body_indexes]
    ),
    dim=-1,
  )
  return _mask_trajectory_envs(command, torch.exp(-error.mean(-1) / std**2))


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = _trajectory_task_command(env, command_name)
  traj = command.trajectory
  body_indexes = _tracking_body_indexes(traj, body_names)
  _, _, ref_body_ang_vel_b, robot_body_ang_vel_b = tracking_body_velocity_b(traj)
  error = torch.sum(
    torch.square(
      ref_body_ang_vel_b[:, body_indexes]
      - robot_body_ang_vel_b[:, body_indexes]
    ),
    dim=-1,
  )
  return _mask_trajectory_envs(command, torch.exp(-error.mean(-1) / std**2))


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions using force history or instantaneous contacts."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    hit = (force_mag > force_threshold).any(dim=1)
    return hit.sum(dim=-1).float()
  assert data.found is not None
  return data.found.squeeze(-1)


@dataclass(frozen=True)
class TrajectoryTrackingRewardCfg:
  """Weights and exponential decay factors for trajectory tracking reward."""

  weights: dict[str, float] = field(
    default_factory=lambda: {
      "position": 3.5,
      "rotation": 2.5,
      "linear_velocity": 0.5,
      "angular_velocity": 0.5,
      "root_height": 1.0,
      "energy": 1.0e-3,
    }
  )
  alphas: dict[str, float] = field(
    default_factory=lambda: {
      "position": 2.5,
      "rotation": 1.0,
      "linear_velocity": 0.12,
      "angular_velocity": 0.05,
      "root_height": 20.0,
    }
  )


def trajectory_tracking_reward_from_errors(
  errors: dict[str, torch.Tensor],
  cfg: TrajectoryTrackingRewardCfg,
  active_mask: torch.Tensor,
) -> torch.Tensor:
  """Compute masked ``sum(w_x * exp(-alpha_x * error_x))`` tracking reward."""
  missing = [name for name in TRACKING_COMPONENTS if name not in errors]
  if missing:
    msg = f"Missing trajectory tracking errors: {missing}"
    raise KeyError(msg)

  first = errors[TRACKING_COMPONENTS[0]]
  reward = torch.zeros_like(first)
  for name in TRACKING_EXP_COMPONENTS:
    reward = reward + cfg.weights[name] * torch.exp(-cfg.alphas[name] * errors[name])
  reward = reward - cfg.weights["energy"] * errors["energy"]
  return torch.where(active_mask, reward, torch.zeros_like(reward))


@dataclass(kw_only=True)
class RecoveryTrajectoryCommandCfg:
  """Configuration for the trajectory child command used by recovery."""

  motion_file: str
  entity_name: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  end_effector_body_names: tuple[str, ...]
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (0.0, 0.0)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  standing_root_height: float = 0.65
  future_velocity_window_steps: int = 30
  future_velocity_min_speed: float = 0.1

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> RecoveryTrajectoryCommand:
    return RecoveryTrajectoryCommand(self, env)


class RecoveryTrajectoryCommand:
  """Masked full-body reference trajectory command.

  This is intentionally not registered with the command manager directly. The
  recovery wrapper owns it and calls it only for trajectory-task environments.
  """

  cfg: RecoveryTrajectoryCommandCfg

  def __init__(self, cfg: RecoveryTrajectoryCommandCfg, env: ManagerBasedRlEnv):
    if not cfg.motion_file:
      msg = "RecoveryTrajectoryCommandCfg.motion_file must point to a trajectory npz."
      raise ValueError(msg)
    self.cfg = cfg
    self._env = env
    self.device = torch.device(env.device)
    self.num_envs = env.num_envs
    self.robot = env.scene[cfg.entity_name]
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.anchor_body_index = self.motion_anchor_body_index
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.end_effector_body_indexes = torch.tensor(
      [cfg.body_names.index(name) for name in cfg.end_effector_body_names],
      dtype=torch.long,
      device=self.device,
    )

    with np.load(cfg.motion_file, allow_pickle=False) as npz:
      missing = [
        key
        for key in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        )
        if key not in npz
      ]
      if missing:
        msg = f"{cfg.motion_file} is missing trajectory keys: {missing}"
        raise KeyError(msg)
      self.motion_joint_pos = torch.as_tensor(
        npz["joint_pos"], dtype=torch.float32, device=self.device
      )
      self.motion_joint_vel = torch.as_tensor(
        npz["joint_vel"], dtype=torch.float32, device=self.device
      )
      body_pos_w = torch.as_tensor(npz["body_pos_w"], dtype=torch.float32, device=self.device)
      body_quat_w = torch.as_tensor(
        npz["body_quat_w"], dtype=torch.float32, device=self.device
      )
      body_lin_vel_w = torch.as_tensor(
        npz["body_lin_vel_w"], dtype=torch.float32, device=self.device
      )
      body_ang_vel_w = torch.as_tensor(
        npz["body_ang_vel_w"], dtype=torch.float32, device=self.device
      )

    self.motion_body_pos_w = body_pos_w[:, self.body_indexes]
    self.motion_body_quat_w = body_quat_w[:, self.body_indexes]
    self.motion_body_lin_vel_w = body_lin_vel_w[:, self.body_indexes]
    self.motion_body_ang_vel_w = body_ang_vel_w[:, self.body_indexes]
    self.motion_root_height = self.motion_body_pos_w[:, 0, 2]
    self.motion_velocity_command = trajectory_velocity_command_from_tensors(
      self.motion_body_quat_w[:, 0],
      self.motion_body_lin_vel_w[:, 0],
      self.motion_body_ang_vel_w[:, 0],
    )
    self.motion_future_standing_velocity_command = (
      future_standing_velocity_commands_from_tensors(
        root_height=self.motion_root_height,
        velocity_command=self.motion_velocity_command,
        standing_height=cfg.standing_root_height,
        window_steps=cfg.future_velocity_window_steps,
      )
    )
    self.time_step_total = int(self.motion_joint_pos.shape[0])
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.bin_count = int(self.time_step_total // (1 / env.step_dt)) + 1
    self.bin_failed_count = torch.zeros(
      self.bin_count,
      dtype=torch.float,
      device=self.device,
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count,
      dtype=torch.float,
      device=self.device,
    )
    self.kernel = torch.tensor(
      [cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)],
      dtype=self.bin_failed_count.dtype,
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()
    self.metrics = {
      "sampling_entropy": torch.zeros(self.num_envs, device=self.device),
      "sampling_top1_prob": torch.zeros(self.num_envs, device=self.device),
      "sampling_top1_bin": torch.zeros(self.num_envs, device=self.device),
    }
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0
    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=-1)

  @property
  def velocity_command(self) -> torch.Tensor:
    return self.motion_velocity_command[self.time_steps]

  @property
  def future_standing_velocity_command(self) -> torch.Tensor:
    return self.motion_future_standing_velocity_command[self.time_steps]

  @property
  def is_reference_root_standing(self) -> torch.Tensor:
    return self.motion_root_height[self.time_steps] >= self.cfg.standing_root_height

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion_joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion_joint_vel[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self.motion_body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion_body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion_body_lin_vel_w[self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion_body_ang_vel_w[self.time_steps]

  def root_goal_command(self, horizon_s: float) -> torch.Tensor:
    """Future root pose goal in the current robot root local frame."""
    horizon_steps = max(1, int(round(float(horizon_s) / self._env.step_dt)))
    future_steps = torch.clamp(
      self.time_steps + horizon_steps,
      max=self.time_step_total - 1,
    )
    origins = self._env.scene.env_origins
    return tracking_future_root_goal_command(
      robot_root_pos_w=self.robot.data.root_link_pos_w,
      robot_root_quat_w=self.robot.data.root_link_quat_w,
      ref_root_pos_w=self.motion_body_pos_w[self.time_steps, 0] + origins,
      ref_root_quat_w=self.motion_body_quat_w[self.time_steps, 0],
      ref_future_root_pos_w=self.motion_body_pos_w[future_steps, 0] + origins,
      ref_future_root_quat_w=self.motion_body_quat_w[future_steps, 0],
    )

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, self.anchor_body_index]

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.body_quat_w[:, self.anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.body_lin_vel_w[:, self.anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.body_ang_vel_w[:, self.anchor_body_index]

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
    if env_ids.numel() == 0:
      return {}
    extras = {}
    for metric_name, metric_value in self.metrics.items():
      extras[metric_name] = torch.mean(metric_value[env_ids]).item()
      metric_value[env_ids] = 0.0
    if self.cfg.sampling_mode == "adaptive":
      self._record_adaptive_failures(env_ids)
    self._resample_command(env_ids)
    self._update_relative_reference()
    return extras

  def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
    self.time_steps[env_ids] = torch.randint(
      0,
      self.time_step_total,
      (len(env_ids),),
      device=self.device,
    )
    self.metrics["sampling_entropy"][:] = 1.0
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    elif self.cfg.sampling_mode == "adaptive":
      self._adaptive_sampling(env_ids)
    else:
      msg = f"Unsupported trajectory sampling_mode: {self.cfg.sampling_mode}"
      raise ValueError(msg)
    self._write_reference_state_to_sim(env_ids)

  def _record_adaptive_failures(self, env_ids: torch.Tensor) -> None:
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      current_bin_index = torch.clamp(
        (self.time_steps * self.bin_count) // max(self.time_step_total, 1),
        0,
        self.bin_count - 1,
      )
      fail_bins = current_bin_index[env_ids][episode_failed]
      self._current_bin_failed[:] = torch.bincount(
        fail_bins,
        minlength=self.bin_count,
      )

  def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
    sampling_probabilities = (
      self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.kernel.numel() - 1),
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities,
      self.kernel.view(1, 1, -1),
    ).view(-1)
    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities,
      len(env_ids),
      replacement=True,
    )
    self.time_steps[env_ids] = (
      (
        sampled_bins
        + sample_uniform(
          0.0,
          1.0,
          (len(env_ids),),
          device=self.device,
        )
      )
      / self.bin_count
      * (self.time_step_total - 1)
    ).long()

    entropy = -(sampling_probabilities * (sampling_probabilities + 1.0e-12).log()).sum()
    normalized_entropy = (
      entropy / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    )
    top_probability, top_bin = sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = normalized_entropy
    self.metrics["sampling_top1_prob"][:] = top_probability
    self.metrics["sampling_top1_bin"][:] = top_bin.float() / self.bin_count

  def compute(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    self.time_steps[env_ids] += 1
    done = env_ids[self.time_steps[env_ids] >= self.time_step_total]
    if done.numel() > 0:
      self._resample_command(done)
    self._update_relative_reference()
    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _update_relative_reference(self) -> None:
    self.body_pos_relative_w, self.body_quat_relative_w = align_reference_body_state(
      ref_body_pos_w=self.body_pos_w,
      ref_body_quat_w=self.body_quat_w,
      ref_anchor_pos_w=self.anchor_pos_w,
      ref_anchor_quat_w=self.anchor_quat_w,
      robot_anchor_pos_w=self.robot_anchor_pos_w,
      robot_anchor_quat_w=self.robot_anchor_quat_w,
    )

  def _write_reference_state_to_sim(self, env_ids: torch.Tensor) -> None:
    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    pose_ranges = torch.tensor(
      [
        self.cfg.pose_range.get(key, (0.0, 0.0))
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      device=self.device,
    )
    pose_samples = sample_uniform(
      pose_ranges[:, 0],
      pose_ranges[:, 1],
      (env_ids.numel(), 6),
      device=self.device,
    )
    root_pos[env_ids] += pose_samples[:, :3]
    root_ori[env_ids] = quat_mul(
      quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
      ),
      root_ori[env_ids],
    )

    velocity_ranges = torch.tensor(
      [
        self.cfg.velocity_range.get(key, (0.0, 0.0))
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      device=self.device,
    )
    velocity_samples = sample_uniform(
      velocity_ranges[:, 0],
      velocity_ranges[:, 1],
      (env_ids.numel(), 6),
      device=self.device,
    )
    root_lin_vel[env_ids] += velocity_samples[:, :3]
    root_ang_vel[env_ids] += velocity_samples[:, 3:]

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )
    joint_pos = self.joint_pos[env_ids].clone()
    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,
    )
    limits = getattr(self.robot.data, "soft_joint_pos_limits", None)
    if limits is not None:
      joint_pos = torch.clip(joint_pos, limits[env_ids, :, 0], limits[env_ids, :, 1])
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.write_joint_state_to_sim(joint_pos, self.joint_vel[env_ids], env_ids=env_ids)
    self.robot.clear_state(env_ids=env_ids)

  def _debug_vis_impl(self, visualizer) -> None:
    """Draw the reference motion like the standalone tracking task."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      indexing = self.robot.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_rotation_matrix = matrix_from_quat(
          self.anchor_quat_w[batch]
        ).cpu().numpy()
        visualizer.add_frame(
          position=self.anchor_pos_w[batch].cpu().numpy(),
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_rotation_matrix = matrix_from_quat(
          self.robot_anchor_quat_w[batch]
        ).cpu().numpy()
        visualizer.add_frame(
          position=self.robot_anchor_pos_w[batch].cpu().numpy(),
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )


class _MaskedDebugVisualizer:
  """Filter visible env ids while delegating draw calls to the real visualizer."""

  def __init__(self, visualizer, active_mask: torch.Tensor):
    self._visualizer = visualizer
    self._active_env_ids = set(active_mask.nonzero(as_tuple=False).flatten().cpu().tolist())

  def get_env_indices(self, num_envs: int) -> list[int]:
    return [
      env_id
      for env_id in self._visualizer.get_env_indices(num_envs)
      if env_id in self._active_env_ids
    ]

  def __getattr__(self, name: str):
    return getattr(self._visualizer, name)


class RecoveryTaskCommand(CommandTerm):
  """Command wrapper that routes envs between trajectory and velocity tasks."""

  cfg: RecoveryTaskCommandCfg

  def __init__(self, cfg: RecoveryTaskCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._task_tensors = build_recovery_task_tensors(
      num_envs=self.num_envs,
      trajectory_fraction=cfg.trajectory_fraction,
      device=self.device,
      trajectory_task_id=cfg.trajectory_task_id,
      velocity_task_id=cfg.velocity_task_id,
    )
    self.trajectory = cfg.trajectory_cfg.build(env)
    self.velocity: CommandTerm = cfg.velocity_cfg.build(env)

  @property
  def is_trajectory_env(self) -> torch.Tensor:
    return self._task_tensors.is_trajectory_env

  @property
  def is_velocity_env(self) -> torch.Tensor:
    return self._task_tensors.is_velocity_env

  @property
  def task_id(self) -> torch.Tensor:
    return self._task_tensors.task_id

  @property
  def velocity_command(self) -> torch.Tensor:
    return compose_recovery_command(
      velocity_command=self.velocity.command,
      trajectory_velocity_command=self.trajectory.velocity_command,
      trajectory_future_velocity_command=(
        self.trajectory.future_standing_velocity_command
      ),
      trajectory_is_standing=self.trajectory.is_reference_root_standing,
      # Old explicit trajectory command kept for rollback:
      # trajectory_command=self.trajectory.command,
      trajectory_command=None,
      is_velocity_env=self.is_velocity_env,
      is_trajectory_env=self.is_trajectory_env,
      trajectory_future_velocity_min_speed=(
        self.trajectory.cfg.future_velocity_min_speed
      ),
    )

  @property
  def command(self) -> torch.Tensor:
    return self.velocity_command

  def compute(self, dt: float) -> None:
    self.trajectory.compute(self.is_trajectory_env.nonzero(as_tuple=False).flatten())
    self.velocity.compute(dt)

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    traj_ids = env_ids[self.is_trajectory_env[env_ids]]
    extras: dict[str, float] = {}
    traj_extras = self.trajectory.reset(traj_ids)
    extras.update({f"traj_{key}": value for key, value in traj_extras.items()})
    vel_extras = self.velocity.reset(env_ids)
    extras.update({f"vel_{key}": value for key, value in vel_extras.items()})
    return extras

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    pass

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer) -> None:
    self.trajectory._debug_vis_impl(
      _MaskedDebugVisualizer(visualizer, self.is_trajectory_env)
    )
    self.velocity._debug_vis_impl(
      _MaskedDebugVisualizer(visualizer, self.is_velocity_env)
    )

  def create_gui(
    self,
    name: str,
    server,
    get_env_idx,
    on_change=None,
    request_action=None,
  ) -> None:
    self.velocity.create_gui(
      f"{name}_velocity",
      server,
      get_env_idx,
      on_change=on_change,
      request_action=request_action,
    )


@dataclass(kw_only=True)
class RecoveryTaskCommandCfg(CommandTermCfg):
  """Configuration for ``RecoveryTaskCommand``."""

  trajectory_cfg: RecoveryTrajectoryCommandCfg
  velocity_cfg: CommandTermCfg
  trajectory_fraction: float = 0.5
  trajectory_task_id: float = TRAJECTORY_TASK_ID
  velocity_task_id: float = VELOCITY_TASK_ID
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> RecoveryTaskCommand:
    return RecoveryTaskCommand(self, env)


class RecoveryGoalCommand(RecoveryTaskCommand):
  """Recovery command wrapper with a local future-root-pose goal observation."""

  cfg: RecoveryGoalCommandCfg

  def __init__(self, cfg: RecoveryGoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.stand_up_speed = torch.zeros(self.num_envs, device=self.device)
    self._resample_stand_up_speed(torch.arange(self.num_envs, device=self.device))

  def _resample_stand_up_speed(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    low, high = self.cfg.stand_up_speed_range
    self.stand_up_speed[env_ids] = torch.empty(
      env_ids.numel(),
      device=self.device,
    ).uniform_(low, high)

  @property
  def command(self) -> torch.Tensor:
    robot = self.trajectory.robot
    trajectory_goal = self.trajectory.root_goal_command(self.cfg.goal_horizon_s)
    velocity_goal = velocity_goal_command(
      velocity_command=self.velocity.command,
      current_root_pos_w=robot.data.root_link_pos_w,
      current_root_quat_w=robot.data.root_link_quat_w,
      root_height=robot.data.root_link_pos_w[:, 2],
      horizon_s=self.cfg.goal_horizon_s,
      standing_height=self.cfg.standing_height,
      stand_up_threshold=self.cfg.stand_up_threshold,
      stand_up_speed=self.stand_up_speed,
    )
    return compose_recovery_goal_command(
      trajectory_goal_command=trajectory_goal,
      velocity_goal_command=velocity_goal,
      is_trajectory_env=self.is_trajectory_env,
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    extras = super().reset(env_ids)
    self._resample_stand_up_speed(env_ids)
    return extras


@dataclass(kw_only=True)
class RecoveryGoalCommandCfg(RecoveryTaskCommandCfg):
  """Configuration for ``RecoveryGoalCommand``."""

  goal_horizon_s: float = 1.0
  standing_height: float = 0.65
  stand_up_threshold: float = 0.65
  stand_up_speed_range: tuple[float, float] = (0.5, 1.2)

  def build(self, env: ManagerBasedRlEnv) -> RecoveryGoalCommand:
    return RecoveryGoalCommand(self, env)


def recovery_task_id(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, RecoveryTaskCommand)
  return command.task_id


def trajectory_tracking_errors_from_tensors(
  *,
  ref_body_pos_w: torch.Tensor,
  ref_body_quat_w: torch.Tensor,
  ref_body_lin_vel_w: torch.Tensor,
  ref_body_ang_vel_w: torch.Tensor,
  cur_body_pos_w: torch.Tensor,
  cur_body_quat_w: torch.Tensor,
  cur_body_lin_vel_w: torch.Tensor,
  cur_body_ang_vel_w: torch.Tensor,
  actuator_force: torch.Tensor,
  joint_vel: torch.Tensor,
  anchor_body_index: int,
  root_body_index: int = 0,
) -> dict[str, torch.Tensor]:
  anchor_pos = cur_body_pos_w[:, anchor_body_index : anchor_body_index + 1].expand_as(
    ref_body_pos_w
  )
  anchor_quat = cur_body_quat_w[:, anchor_body_index : anchor_body_index + 1].expand_as(
    ref_body_quat_w
  )
  ref_pos_local, ref_quat_local = subtract_frame_transforms(
    anchor_pos,
    anchor_quat,
    ref_body_pos_w,
    ref_body_quat_w,
  )
  cur_pos_local, cur_quat_local = subtract_frame_transforms(
    anchor_pos,
    anchor_quat,
    cur_body_pos_w,
    cur_body_quat_w,
  )
  ref_body_lin_vel_local = quat_apply_inverse(anchor_quat, ref_body_lin_vel_w)
  cur_body_lin_vel_local = quat_apply_inverse(anchor_quat, cur_body_lin_vel_w)
  ref_body_ang_vel_local = quat_apply_inverse(anchor_quat, ref_body_ang_vel_w)
  cur_body_ang_vel_local = quat_apply_inverse(anchor_quat, cur_body_ang_vel_w)
  n_power_terms = min(actuator_force.shape[1], joint_vel.shape[1])

  return {
    "position": torch.linalg.norm(ref_pos_local - cur_pos_local, dim=-1).mean(dim=-1),
    "rotation": quat_error_magnitude(ref_quat_local, cur_quat_local).mean(dim=-1),
    "linear_velocity": torch.linalg.norm(
      ref_body_lin_vel_local - cur_body_lin_vel_local, dim=-1
    ).mean(dim=-1),
    "angular_velocity": torch.linalg.norm(
      ref_body_ang_vel_local - cur_body_ang_vel_local, dim=-1
    ).mean(dim=-1),
    "root_height": torch.abs(
      ref_pos_local[:, root_body_index, 2] - cur_pos_local[:, root_body_index, 2]
    ),
    "energy": torch.sum(
      torch.abs(actuator_force[:, :n_power_terms] * joint_vel[:, :n_power_terms]),
      dim=-1,
    ),
  }


def trajectory_tracking_errors(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> dict[str, torch.Tensor]:
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, RecoveryTaskCommand)
  traj = command.trajectory
  ref_body_pos_b, ref_body_quat_b, robot_body_pos_b, robot_body_quat_b = (
    tracking_body_pose_b(traj)
  )
  ref_body_lin_vel_b, robot_body_lin_vel_b, ref_body_ang_vel_b, robot_body_ang_vel_b = (
    tracking_body_velocity_b(traj)
  )

  num_bodies = len(traj.cfg.body_names)
  anchor_pos_w, anchor_quat_w = _repeat_anchor_frame(
    traj.robot_anchor_pos_w,
    traj.robot_anchor_quat_w,
    num_bodies,
  )
  ref_body_pos_local, _ = subtract_frame_transforms(
    anchor_pos_w,
    anchor_quat_w,
    traj.body_pos_w,
    traj.body_quat_w,
  )
  robot_body_pos_local, _ = subtract_frame_transforms(
    anchor_pos_w,
    anchor_quat_w,
    traj.robot_body_pos_w,
    traj.robot_body_quat_w,
  )
  n_power_terms = min(
    traj.robot.data.actuator_force.shape[1],
    traj.robot.data.joint_vel.shape[1],
  )

  return {
    "position": torch.linalg.norm(
      ref_body_pos_b - robot_body_pos_b, dim=-1
    ).mean(dim=-1),
    "rotation": quat_error_magnitude(ref_body_quat_b, robot_body_quat_b).mean(
      dim=-1
    ),
    "linear_velocity": torch.linalg.norm(
      ref_body_lin_vel_b - robot_body_lin_vel_b, dim=-1
    ).mean(dim=-1),
    "angular_velocity": torch.linalg.norm(
      ref_body_ang_vel_b - robot_body_ang_vel_b, dim=-1
    ).mean(dim=-1),
    "root_height": torch.abs(
      ref_body_pos_local[:, 0, 2] - robot_body_pos_local[:, 0, 2]
    ),
    "energy": torch.sum(
      torch.abs(
        traj.robot.data.actuator_force[:, :n_power_terms]
        * traj.robot.data.joint_vel[:, :n_power_terms]
      ),
      dim=-1,
    ),
  }


def trajectory_tracking_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  tracking_cfg: TrajectoryTrackingRewardCfg | None = None,
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, RecoveryTaskCommand)
  if tracking_cfg is None:
    tracking_cfg = TrajectoryTrackingRewardCfg()
  return trajectory_tracking_reward_from_errors(
    trajectory_tracking_errors(env, command_name),
    tracking_cfg,
    command.is_trajectory_env,
  )


def masked_velocity_smp_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  task_terms: tuple[TaskTerm, ...],
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
  subsample_steps: int = 1,
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, RecoveryTaskCommand)
  reward = task_smp_product(
    env,
    task_terms=task_terms,
    fixed_timesteps=fixed_timesteps,
    ws=ws,
    command_name=command_name,
    subsample_steps=subsample_steps,
  )
  return torch.where(command.is_velocity_env, reward, torch.zeros_like(reward))
