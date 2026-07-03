"""Velocity-tracking reward components for the localization task.

All terms are SMP-gated via the generic ``smp.rl.rewards.task_smp_product``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  reverse_penalty: bool = False,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.

  When ``reverse_penalty=True``, the reward is zeroed whenever the body-frame
  velocity projects negatively onto the commanded direction (i.e. the robot is
  moving the wrong way).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  reward = torch.exp(-lin_vel_error / std**2)
  if reverse_penalty:
    proj_speed = (command[:, :2] * actual[:, :2]).sum(dim=-1)
    reward = torch.where(proj_speed < 0, torch.zeros_like(reward), reward)
  return reward


def track_velocity(
  env: ManagerBasedRlEnv,
  lin_std: float,
  ang_std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the full body-frame twist command ``[vx, vy, ω]``.

  Linear error: ``‖cmd[:2] − v_b[:2]‖² + v_b.z²`` (command z is assumed 0).
  Angular error: ``(cmd[2] − ω_b.z)² + ω_b.xy²`` (command xy is assumed 0).

  Returns ``exp(-lin_err/lin_std²) * exp(-ang_err/ang_std²)`` so both
  components must be good for a high reward.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  # Linear velocity (body-frame).
  actual_lin = asset.data.root_link_lin_vel_b
  lin_xy_err = torch.sum(torch.square(command[:, :2] - actual_lin[:, :2]), dim=1)
  lin_z_err = torch.square(actual_lin[:, 2])
  lin_err = lin_xy_err + lin_z_err

  # Angular velocity (body-frame).
  actual_ang = asset.data.root_link_ang_vel_b
  ang_z_err = torch.square(command[:, 2] - actual_ang[:, 2])
  ang_xy_err = torch.sum(torch.square(actual_ang[:, :2]), dim=1)
  ang_err = ang_z_err + ang_xy_err

  return torch.exp(-lin_err / lin_std**2) * torch.exp(-ang_err / ang_std**2)


def track_linear_velocity_world(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity, **evaluated in the
  world frame** so the error signal aligns with the SMP motion prior.

  The body-frame command ``[vx, vy]`` is rotated into the world frame via the
  robot's current heading before comparing against the world-frame root velocity.
  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  heading_w = asset.data.heading_w
  cos_h = torch.cos(heading_w)
  sin_h = torch.sin(heading_w)

  # Rotate body-frame command → world frame.
  cmd_x_w = cos_h * command[:, 0] - sin_h * command[:, 1]
  cmd_y_w = sin_h * command[:, 0] + cos_h * command[:, 1]

  actual_w = asset.data.root_link_lin_vel_w
  xy_error = torch.square(cmd_x_w - actual_w[:, 0]) + torch.square(cmd_y_w - actual_w[:, 1])
  z_error = torch.square(actual_w[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)

def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward

def phase(
    env: ManagerBasedRlEnv,
    period: float,
    command_name: str = "",
    command_threshold: float = 0.1,
) -> torch.Tensor:
  """Gait phase observation: ``[sin(2π·φ), cos(2π·φ)]`` based on episode time.

  When ``command_name`` is given, environments whose command magnitude falls
  below ``command_threshold`` receive ``[0.0, 0.0]`` so the policy does not
  try to follow a gait cycle during standing.
  """
  t = env.episode_length_buf.float() * env.step_dt
  phi = (t / period) % 1.0
  sin_phase = torch.sin(2 * torch.pi * phi)
  cos_phase = torch.cos(2 * torch.pi * phi)
  obs = torch.stack([sin_phase, cos_phase], dim=-1)
  if command_name:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      obs[cmd_norm <= command_threshold] = 0.0
  return obs


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


def _head_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  return robot.data.site_pos_w[:, head_idx, 2]

def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize action changes (L2 norm of action difference)."""
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )

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


def upward_velocity_when_low(
  env: ManagerBasedRlEnv,
  target_velocity: float = 0.25,
  head_height_threshold: float = 0.6,
  scale: float = 100.0,
  upright_threshold: float = 0.9,
) -> torch.Tensor:
  """``upward_velocity`` gated: zeroed when head ≥ ``upright_threshold``.

  The original ``upward_velocity`` logic (including its internal
  ``head_height_threshold`` gating that returns 1 above threshold) is
  preserved.  Above ``upright_threshold`` the reward is zeroed so it
  yields to velocity-tracking terms.
  """
  reward = upward_velocity(env, target_velocity, head_height_threshold, scale)
  head_z = _head_height(env)
  return torch.where(head_z < upright_threshold, reward, torch.zeros_like(reward))


def track_linear_velocity_upright(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  reverse_penalty: bool = False,
  head_height_threshold: float = 0.9,
) -> torch.Tensor:
  """``track_linear_velocity`` gated: active only when head ≥ threshold.

  Below the threshold this returns zero so the policy focuses on getting up
  rather than tracking velocity commands it cannot satisfy yet.
  """
  reward = track_linear_velocity(env, std, command_name, asset_cfg, reverse_penalty)
  head_z = _head_height(env)
  return torch.where(head_z >= head_height_threshold, reward, torch.zeros_like(reward))


def track_angular_velocity_upright(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  head_height_threshold: float = 0.9,
) -> torch.Tensor:
  """``track_angular_velocity`` gated: active only when head ≥ threshold.

  Below the threshold this returns zero so the policy focuses on getting up
  rather than tracking angular velocity commands.
  """
  reward = track_angular_velocity(env, std, command_name, asset_cfg)
  head_z = _head_height(env)
  return torch.where(head_z >= head_height_threshold, reward, torch.zeros_like(reward))


def track_head_height_when_low(
  env: ManagerBasedRlEnv,
  target_height: float = 1.2,
  scale: float = 6.0,
  head_height_threshold: float = 0.9,
) -> torch.Tensor:
  """``track_head_height`` gated: active only when head **<** threshold.

  Above the threshold the robot is already upright — the height reward
  is no longer needed and yields to the velocity-tracking terms.
  """
  reward = track_head_height(env, target_height, scale)
  head_z = _head_height(env)
  return torch.where(head_z < head_height_threshold, reward, torch.zeros_like(reward))
