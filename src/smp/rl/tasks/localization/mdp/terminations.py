"""Termination terms for the localization task."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def root_height_below_minimum(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  grace_steps: int = 15,
) -> torch.Tensor:
  """Terminate when root height is below minimum after the grace period."""
  asset = env.scene[asset_cfg.name]
  below_minimum = asset.data.root_link_pos_w[:, 2] < minimum_height
  past_grace = env.episode_length_buf >= grace_steps
  return below_minimum & past_grace

# 脱离了合理的运动空间，就立即强制结束当前的训练回合
def smp_too_low(
  env: ManagerBasedRlEnv,
  threshold: float = 0.02,
  ws: float = 6.0,
  grace_steps: int = 15,
) -> torch.Tensor:
  """Terminate when the SMP score collapses (off-manifold): end if
  ``exp(-ws·env._smp_raw_err) < threshold`` past ``grace_steps``.  Uses the RAW MSE
  (stable absolute scale), so ``ws`` must match the reward's.  Kills the "violent
  get-up" shortcut — leaving the manifold drives the score to 0."""
  raw_err = getattr(env, "_smp_raw_err", None)
  if raw_err is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  raw_smp = torch.exp(-ws * raw_err)
  past_grace = env.episode_length_buf >= grace_steps
  return (raw_smp < threshold) & past_grace


# 判断机器人是否已经成功站立，并且要求这种站立是“稳定”的（达到一定高度、速度极小，且能保持足够长的时间）。一旦达成，就结束当前回合
# 通过将成功标记为 Truncation（在大多数框架中通过传 time_out=True 或类似的 flag 实现），
# RL 算法的 Critic 网络会进行自举。它会告诉 AI：“回合虽然结束了，但如果你继续保持这个站立姿势，未来还能拿到源源不断的奖励（$V(s_{t+1})$）”
def stood_up(
  env: ManagerBasedRlEnv,
  head_height: float = 1.2,
  max_speed: float = 0.5,
  hold_steps: int = 10,
) -> torch.Tensor:
  """Truncate after recovering from below standing height.

  The velocity task can start from an already-upright pose, so stable standing
  only counts as success after this episode has first gone below ``head_height``.
  Wire ``time_out=True`` so it's a TRUNCATION — the value bootstraps from the
  standing state, else standing looks worthless.
  """
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  z = robot.data.site_pos_w[:, head_idx, 2]
  speed = torch.linalg.norm(robot.data.root_link_lin_vel_w, dim=-1)
  is_standing = (z >= head_height) & (speed < max_speed)

  cnt = getattr(env, "_localization_stand_count", None)
  if cnt is None:
    cnt = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  was_below = getattr(env, "_localization_was_below_stand_height", None)
  if was_below is None:
    was_below = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  episode_length = getattr(env, "episode_length_buf", None)
  if episode_length is not None:
    new_episode = episode_length <= 1
    cnt = torch.where(new_episode, torch.zeros_like(cnt), cnt)
    was_below = torch.where(new_episode, torch.zeros_like(was_below), was_below)

  was_below = was_below | (z < head_height)
  cnt = torch.where(is_standing, cnt + 1, torch.zeros_like(cnt))
  env._localization_stand_count = cnt  # type: ignore[attr-defined]
  env._localization_was_below_stand_height = was_below  # type: ignore[attr-defined]
  return was_below & (cnt >= hold_steps)
