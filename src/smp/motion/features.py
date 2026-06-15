"""Shared SMP motion feature construction.

Input tensors are windows of raw kinematic state:
``root_pos``, ``root_quat``, ``root_lin_vel``, ``root_ang_vel``, ``ee_pos``,
and ``joint_pos``.  Output layout matches the diffusion prior training data:

  [root_pos(3), root_rot(6), joint_pos(J), ee_pos(E*3),
   root_lin_vel(3), root_ang_vel(3)]

All spatial quantities are expressed in the LAST frame's yaw-only local frame.
"""

from __future__ import annotations

import torch

from smp.motion.math import (
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  tan_norm_from_quat,
  yaw_quat,
)

NUM_JOINTS = 29

EE_BODY_NAMES: tuple[str, ...] = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)
NUM_EE = len(EE_BODY_NAMES)

FEATURE_DIMS: tuple[int, ...] = (3, 6, NUM_JOINTS, NUM_EE * 3, 3, 3)
FEATURE_DIM = sum(FEATURE_DIMS)


def compute_motion_features(
  *,
  root_pos: torch.Tensor,
  root_quat: torch.Tensor,
  root_lin_vel: torch.Tensor,
  root_ang_vel: torch.Tensor,
  ee_pos: torch.Tensor,
  joint_pos: torch.Tensor,
) -> torch.Tensor:
  """Build SMP diffusion features from raw windowed kinematic state.

  Args:
    root_pos: ``(N, W, 3)`` root positions.
    root_quat: ``(N, W, 4)`` wxyz root quaternions.
    root_lin_vel: ``(N, W, 3)`` root linear velocities.
    root_ang_vel: ``(N, W, 3)`` root angular velocities.
    ee_pos: ``(N, W, E, 3)`` end-effector positions.
    joint_pos: ``(N, W, J)`` raw joint positions.
  """
  if root_pos.ndim != 3:
    msg = f"root_pos must have shape (N, W, 3); got {tuple(root_pos.shape)}"
    raise ValueError(msg)
  if root_quat.ndim != 3:
    msg = f"root_quat must have shape (N, W, 4); got {tuple(root_quat.shape)}"
    raise ValueError(msg)
  if root_lin_vel.ndim != 3:
    msg = f"root_lin_vel must have shape (N, W, 3); got {tuple(root_lin_vel.shape)}"
    raise ValueError(msg)
  if root_ang_vel.ndim != 3:
    msg = f"root_ang_vel must have shape (N, W, 3); got {tuple(root_ang_vel.shape)}"
    raise ValueError(msg)
  if ee_pos.ndim != 4:
    msg = f"ee_pos must have shape (N, W, E, 3); got {tuple(ee_pos.shape)}"
    raise ValueError(msg)
  if joint_pos.ndim != 3:
    msg = f"joint_pos must have shape (N, W, J); got {tuple(joint_pos.shape)}"
    raise ValueError(msg)
  n, window_size, _ = root_pos.shape
  num_ee = ee_pos.shape[2]

  anchor_pos_t = root_pos[:, -1]
  anchor_quat_t = root_quat[:, -1]
  yaw_t = yaw_quat(anchor_quat_t)
  heading_inv_t_w = quat_conjugate(yaw_t)[:, None, :].expand(n, window_size, 4)
  yaw_t_w = yaw_t[:, None, :].expand(n, window_size, 4).reshape(-1, 4)

  root_offset = root_pos - anchor_pos_t[:, None, :]
  root_pos_local = quat_apply_inverse(yaw_t_w, root_offset.reshape(-1, 3)).reshape(
    n, window_size, 3
  )
  root_pos_local = root_pos_local.clone()
  root_pos_local[..., 2] = root_pos[..., 2]

  root_rot_local_quat = quat_mul(
    heading_inv_t_w.reshape(-1, 4),
    root_quat.reshape(-1, 4),
  ).reshape(n, window_size, 4)
  root_rot_6d = tan_norm_from_quat(root_rot_local_quat)

  ee_offset_w = ee_pos - root_pos[:, :, None, :]
  yaw_t_e = yaw_t[:, None, None, :].expand(n, window_size, num_ee, 4).reshape(-1, 4)
  ee_pos_local = quat_apply_inverse(yaw_t_e, ee_offset_w.reshape(-1, 3)).reshape(
    n, window_size, num_ee * 3
  )

  lin_vel_local = quat_apply_inverse(yaw_t_w, root_lin_vel.reshape(-1, 3)).reshape(
    n, window_size, 3
  )
  ang_vel_local = quat_apply_inverse(yaw_t_w, root_ang_vel.reshape(-1, 3)).reshape(
    n, window_size, 3
  )

  return torch.cat(
    [
      root_pos_local,
      root_rot_6d,
      joint_pos,
      ee_pos_local,
      lin_vel_local,
      ang_vel_local,
    ],
    dim=-1,
  )


def slice_motion_features(frame: torch.Tensor) -> dict[str, torch.Tensor]:
  """Slice SMP feature vectors into named components."""
  if frame.shape[-1] != FEATURE_DIM:
    msg = f"expected feature_dim={FEATURE_DIM}; got {frame.shape[-1]}"
    raise ValueError(msg)
  joint_pos_end = 9 + NUM_JOINTS
  ee_pos_end = joint_pos_end + NUM_EE * 3
  lin_vel_end = ee_pos_end + 3
  ang_vel_end = lin_vel_end + 3
  return {
    "root_pos": frame[..., 0:3],
    "root_rot": frame[..., 3:9],
    "joint_pos": frame[..., 9:joint_pos_end],
    "ee_pos": frame[..., joint_pos_end:ee_pos_end],
    "root_lin_vel": frame[..., ee_pos_end:lin_vel_end],
    "root_ang_vel": frame[..., lin_vel_end:ang_vel_end],
  }
