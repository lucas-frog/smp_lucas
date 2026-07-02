"""SMP localization tasks — registers velocity variants on import."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.localization.velocity_env_cfg import g1_velocity_composed_smp_env_cfg
from smp.rl.tasks.localization.velocity_env_cfg import g1_velocity_single_style_smp_env_cfg
from smp.rl.tasks.localization.velocity_env_cfg import g1_velocity_smp_env_cfg
from smp.rl.tasks.localization.velocity_env_cfg import g1_velocity_walkrun_product_smp_env_cfg

_velocity_rl = unitree_g1_smp_ppo_runner_cfg()
_velocity_rl.experiment_name = "smp_velocity_g1"
_velocity_rl.run_name = "smp_velocity_g1"

register_mjlab_task(
  task_id="Smp-Velocity-G1",
  env_cfg=g1_velocity_smp_env_cfg(play=False),
  play_env_cfg=g1_velocity_smp_env_cfg(play=True),
  rl_cfg=_velocity_rl,
)

_velocity_sum_rl = unitree_g1_smp_ppo_runner_cfg()
_velocity_sum_rl.experiment_name = "smp_velocity_sum_g1"
_velocity_sum_rl.run_name = "smp_velocity_sum_g1"

register_mjlab_task(
  task_id="Smp-Velocity-Sum-G1",
  env_cfg=g1_velocity_smp_env_cfg(
    play=False, reward_mode="sum", task_scale=0.5, smp_scale=1.5
  ),
  play_env_cfg=g1_velocity_smp_env_cfg(
    play=True, reward_mode="sum", task_scale=0.5, smp_scale=1.5
  ),
  rl_cfg=_velocity_sum_rl,
)

_velocity_composed_rl = unitree_g1_smp_ppo_runner_cfg()
_velocity_composed_rl.experiment_name = "smp_velocity_composed_g1"
_velocity_composed_rl.run_name = "smp_velocity_composed_g1"

register_mjlab_task(
  task_id="Smp-Velocity-Composed-G1",
  env_cfg=g1_velocity_composed_smp_env_cfg(play=False),
  play_env_cfg=g1_velocity_composed_smp_env_cfg(play=True),
  rl_cfg=_velocity_composed_rl,
)

_velocity_single_style_rl = unitree_g1_smp_ppo_runner_cfg()
_velocity_single_style_rl.experiment_name = "smp_velocity_single_style_g1"
_velocity_single_style_rl.run_name = "smp_velocity_single_style_g1"

register_mjlab_task(
  task_id="Smp-Velocity-SingleStyle-G1",
  env_cfg=g1_velocity_single_style_smp_env_cfg(play=False),
  play_env_cfg=g1_velocity_single_style_smp_env_cfg(play=True),
  rl_cfg=_velocity_single_style_rl,
)

_velocity_walkrun_product_rl = unitree_g1_smp_ppo_runner_cfg()
_velocity_walkrun_product_rl.experiment_name = "smp_velocity_walkrun_product_g1"
_velocity_walkrun_product_rl.run_name = "smp_velocity_walkrun_product_g1"

register_mjlab_task(
  task_id="Smp-Velocity-SingleStyle-Walkrun-Product-G1",
  env_cfg=g1_velocity_walkrun_product_smp_env_cfg(play=False),
  play_env_cfg=g1_velocity_walkrun_product_smp_env_cfg(play=True),
  rl_cfg=_velocity_walkrun_product_rl,
)

__all__ = [
  "g1_velocity_smp_env_cfg",
  "g1_velocity_composed_smp_env_cfg",
  "g1_velocity_single_style_smp_env_cfg",
  "g1_velocity_walkrun_product_smp_env_cfg",
]
