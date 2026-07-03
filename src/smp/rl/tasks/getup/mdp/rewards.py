"""Reward components for the getup task: head-height + up-velocity.

Combined and SMP-gated via the generic ``smp.rl.rewards.smp_product``.
"""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

__all__ = [
  "action_rate_l2",
  "joint_pos_limits",
  "track_head_height",
  "upward_velocity",
]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _head_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  return robot.data.site_pos_w[:, head_idx, 2]


def action_rate_l2(
  env: ManagerBasedRlEnv,
  head_height_threshold: float = 0.9,
  low_height_scale: float = 0.2,
  high_height_scale: float = 1.0,
) -> torch.Tensor:
  """Penalize action changes with separate low/high head-height scales.

  The low-height scale keeps crouched/low-head jitter from being free while
  still allowing faster action changes during recovery than in upright tracking.
  """
  action_rate = torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )
  head_z = _head_height(env)
  scale = torch.where(
    head_z >= head_height_threshold,
    torch.full_like(action_rate, high_height_scale),
    torch.full_like(action_rate, low_height_scale),
  )
  return action_rate * scale


def track_head_height(
  env: ManagerBasedRlEnv,
  target_height: float = 1.2,
  scale: float = 6.0,
) -> torch.Tensor:
  """Reward the ``head`` site reaching ``target_height``:
  ``exp(-scale·max(target_height − head_z, 0)²)`` (no penalty for overshoot).
  Needs the ``head`` site from ``getup_env_cfg.get_g1_spec_with_head``."""
  z = _head_height(env)
  shortfall = torch.clamp(z - target_height, max=0.0)
  return torch.exp(-scale * shortfall * shortfall)


# 鼓励“头部”以目标速度向上运动；而当头部高度达到一定阈值后，停止施加这个速度要求
def upward_velocity(
  env: ManagerBasedRlEnv,
  target_velocity: float = 0.25,
  head_height_threshold: float = 0.6,
  scale: float = 100.0,
) -> torch.Tensor:
  """Reward upward HEAD velocity below ``head_height_threshold`` (else ``1``):
  ``exp(-scale·max(target_velocity − head_vz, 0)²)``.  Uses the head site's world
  velocity (``site_lin_vel_w``, includes ω×r from torso pitch) so it drives the
  head, not the pelvis.  Needs ``getup_env_cfg.get_g1_spec_with_head``."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  head_vz = robot.data.site_lin_vel_w[:, head_idx, 2]
  shortfall = torch.clamp(head_vz - target_velocity, max=0.0)
  shaped = torch.exp(-scale * shortfall * shortfall)
  return torch.where(
    head_z < head_height_threshold,
    shaped,
    torch.ones_like(shaped),
  )


def joint_pos_limits(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Penalize joint positions if they cross the soft limits."""
  asset: Entity = env.scene[asset_cfg.name]
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
  return torch.sum(out_of_limits, dim=1)
