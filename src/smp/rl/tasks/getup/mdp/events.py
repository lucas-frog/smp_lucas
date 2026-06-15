"""Reset events for the getup task."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = ["reset_stand_counter"]


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
