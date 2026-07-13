"""SMP localization tasks."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.localization.recovery_env_cfg import g1_recovery_smp_env_cfg
from smp.rl.tasks.localization.velocity_env_cfg import g1_velocity_smp_env_cfg

_velocity_rl = unitree_g1_smp_ppo_runner_cfg()
_velocity_rl.experiment_name = "smp_velocity_g1"
_velocity_rl.run_name = "smp_velocity_g1"

_recovery_rl = unitree_g1_smp_ppo_runner_cfg()
_recovery_rl.experiment_name = "smp_recovery_g1"
_recovery_rl.run_name = "smp_recovery_g1"

register_mjlab_task(
  task_id="Smp-Velocity-G1",
  env_cfg=g1_velocity_smp_env_cfg(play=False),
  play_env_cfg=g1_velocity_smp_env_cfg(play=True),
  rl_cfg=_velocity_rl,
)

register_mjlab_task(
  task_id="Smp-Recovery-G1",
  env_cfg=g1_recovery_smp_env_cfg(play=False),
  play_env_cfg=g1_recovery_smp_env_cfg(play=True),
  rl_cfg=_recovery_rl,
)

__all__ = [
  "g1_recovery_smp_env_cfg",
  "g1_velocity_smp_env_cfg",
]
