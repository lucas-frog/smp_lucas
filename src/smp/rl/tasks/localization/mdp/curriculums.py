from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])


def velocity_cmd_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  reward_term_names: list[str] | None = None,
  threshold_ratio: float = 0.8,
  delta: float = 0.1,
) -> dict[str, torch.Tensor]:
  """Adaptively expand velocity command ranges when tracking reward is high.

  Mirrors the Isaac Lab ``lin_vel_cmd_levels`` / ``ang_vel_cmd_levels`` pattern:
  every episode boundary the time-averaged *reward_term_names* are checked; if
  each one exceeds ``threshold_ratio * its_weight`` the linear and angular velocity
  ranges are symmetrically expanded by *delta* (clamped to ``limit_ranges``).

  When *reward_term_names* is ``None``, defaults to ``["task_smp_product"]``.

  Set ``limit_ranges`` on the command config to cap expansion; when absent the
  initial ``ranges`` are used as the ceiling.
  """
  _ = env_ids  # Operates globally on the command config; env_ids unused.

  if reward_term_names is None:
    reward_term_names = ["task_smp_product"]

  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

  # Resolve limit (default: initial ranges — no expansion).
  if cfg.limit_ranges is not None:
    lim_x = cfg.limit_ranges.lin_vel_x
    lim_y = cfg.limit_ranges.lin_vel_y
    lim_z = cfg.limit_ranges.ang_vel_z
  else:
    lim_x = cfg.ranges.lin_vel_x
    lim_y = cfg.ranges.lin_vel_y
    lim_z = cfg.ranges.ang_vel_z

  # Average the per-term normalised reward: each term ∈ [0, 1] after dividing
  # by its max possible per-step contribution (weight).
  normalised = torch.zeros(1, device=env.device)
  for name in reward_term_names:
    reward_term = env.reward_manager.get_term_cfg(name)
    avg = (
      torch.mean(env.reward_manager._episode_sums[name][env_ids])
      / env.max_episode_length_s
    )
    w = abs(reward_term.weight) if reward_term.weight != 0.0 else 1.0
    normalised += avg / w
  normalised = normalised / len(reward_term_names)

  if env.common_step_counter % env.max_episode_length == 0:
    if normalised > threshold_ratio:
      d = torch.tensor([-delta, delta], device=env.device)
      cfg.ranges.lin_vel_x = tuple(
        torch.clamp(
          torch.tensor(cfg.ranges.lin_vel_x, device=env.device) + d,
          lim_x[0],
          lim_x[1],
        ).tolist()
      )
      cfg.ranges.lin_vel_y = tuple(
        torch.clamp(
          torch.tensor(cfg.ranges.lin_vel_y, device=env.device) + d,
          lim_y[0],
          lim_y[1],
        ).tolist()
      )
      cfg.ranges.ang_vel_z = tuple(
        torch.clamp(
          torch.tensor(cfg.ranges.ang_vel_z, device=env.device) + d,
          lim_z[0],
          lim_z[1],
        ).tolist()
      )

  return {
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1], device=env.device),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0], device=env.device),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1], device=env.device),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1], device=env.device),
  }
