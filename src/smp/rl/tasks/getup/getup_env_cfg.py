"""G1 getup task with SMP guidance."""

from __future__ import annotations

import mujoco
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec as _get_g1_spec
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.getup import mdp

# Matches the existing ``head_collision`` geom on ``torso_link`` in g1.xml.
HEAD_POS_IN_TORSO: tuple[float, float, float] = (0.0, 0.0, 0.43)

# 添加一个无质量的头部参考点
def get_g1_spec_with_head() -> mujoco.MjSpec:  # type: ignore[attr-defined]
  """Stock G1 spec with a massless ``head`` site on ``torso_link``."""
  spec = _get_g1_spec()
  torso = spec.body("torso_link")
  if not any(s.name == "head" for s in torso.sites):
    torso.add_site(name="head", pos=HEAD_POS_IN_TORSO)
  return spec


def g1_getup_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 getup env cfg with SMP guidance."""
  cfg = g1_smp_env_cfg(play=play)

  # --- Scene ---------------------------------------------------------------
  cfg.scene.entities["robot"].spec_fn = get_g1_spec_with_head

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "logs/pretrain/pretrain/20260614_142121/pretrained.pt"
  )
  cfg.events["reset_stand_counter"] = EventTermCfg(
    func=mdp.reset_stand_counter, mode="reset"
  )

  # --- Rewards -------------------------------------------------------------
  # task = 0.7·upward_velocity + 0.3·head_height, gated by SMP.
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "task_terms": (
        (
          mdp.upward_velocity,
          0.7,
          {
            "target_velocity": 0.25,
            "head_height_threshold": 0.9,
            "scale": 100.0,
          },
        ),
        (mdp.track_head_height, 0.3, {"target_height": 1.1, "scale": 1.0}),
      ),
    },
  )

  # --- Terminations --------------------------------------------------------
  cfg.terminations.pop("self_collision", None)
  cfg.terminations["smp_too_low"] = TerminationTermCfg(
    func=mdp.smp_too_low,
    params={"threshold": 0.02, "ws": 6.0, "grace_steps": 5},
  )

  cfg.terminations["stood_up"] = TerminationTermCfg(
    func=mdp.stood_up,
    time_out=True,
    params={"head_height": 1.2, "max_speed": 0.5, "hold_steps": 25},
  )

  cfg.episode_length_s = 5

  return cfg
