"""G1 velocity task — body-frame twist commands with world-frame reward.

A velocity-tracking task where the policy receives body-frame twist commands
(``[vx, vy, ω]``) and is rewarded via world-frame linear velocity error so the
task signal aligns with the SMP motion prior.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.localization import mdp

import math

def g1_velocity_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 forward env cfg with SMP guidance."""
  cfg = g1_smp_env_cfg(play=play)

  # --- Sensors --------------------------------------------------------------
  # Contact sensor for foot-ground gait rewards (P=2: left / right ankle).
  # feet_ground_contact_cfg = ContactSensorCfg(
  #   name="feet_ground_contact",
  #   primary=ContactMatch(
  #     mode="body",
  #     pattern=r"^(left|right)_ankle_roll_link$",
  #     entity="robot",
  #   ),
  #   secondary=ContactMatch(
  #     mode="body",
  #     pattern="terrain",
  #   ),
  #   fields=("found",),
  #   reduce="none",
  #   num_slots=1,
  #   track_air_time=True,
  # )
  # cfg.scene.sensors += (feet_ground_contact_cfg,)

  # --- Commands ------------------------------------------------------------
  ##
  # Two-phase velocity command: directional categories first (learn basic
  # gaits), then uniform resampling (generalize to full velocity space).
  #
  # phase_switch_iteration: training iteration at which the switch happens.
  #   Set to 0 to use uniform from the start (skip directional phase).
  # steps_per_iteration: must match num_steps_per_env in rl_cfg.py (currently 24).
  ##
  # cfg.commands["twist"] = mdp.PhasedVelocityCommandCfg(
  #   directional_cfg=mdp.DirectionalVelocityCommandCfg(
  #     entity_name="robot",
  #     rel_forward=0.35,
  #     rel_backward=0.35,
  #     rel_left=0.10,
  #     rel_right=0.10,
  #     rel_standing=0.10,
  #     rel_ang_overlay=0.40,
  #     forward_speed=(0.5, 4.0),
  #     backward_speed=(0.5, 2.0),
  #     lateral_speed=(0.3, 0.8),
  #     ang_speed=(0.3, 0.6),
  #     debug_vis=False,  # wrapper handles debug_vis
  #   ),
  #   uniform_cfg=mdp.UniformVelocityCommandCfg(
  #     entity_name="robot",
  #     resampling_time_range=(4.0, 8.0),
  #     rel_standing_envs=0.1,
  #     rel_heading_envs=0.0,
  #     rel_forward_envs=0.0,
  #     heading_command=False,
  #     heading_control_stiffness=0.5,
  #     debug_vis=False,  # wrapper handles debug_vis
  #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
  #       lin_vel_x=(-2.0, 4.0),
  #       lin_vel_y=(-1.0, 1.0),
  #       ang_vel_z=(-0.8, 0.8),
  #     ),
  #   ),
  #   phase_switch_iteration=5000,
  #   steps_per_iteration=24,
  #   debug_vis=True,
  # )

  # --- Legacy single-command configs (kept for reference) ------------------
  #
  # # Directional-only: fixed-per-episode directional categories.
  # cfg.commands["twist"] = mdp.DirectionalVelocityCommandCfg(
  #   entity_name="robot",
  #   rel_forward=0.35, rel_backward=0.35,
  #   rel_left=0.10, rel_right=0.10,
  #   rel_standing=0.10,
  #   rel_ang_overlay=0.40,
  #   forward_speed=(0.5, 4.0),
  #   backward_speed=(0.5, 2.0),
  #   lateral_speed=(0.3, 0.8),
  #   ang_speed=(0.3, 0.6),
  #   debug_vis=True,
  # )
  #
  # # GSI-reset: mirror GSI velocity, hold constant.
  # cfg.commands["twist"] = mdp.GSIResetVelocityCommandCfg(
  #   entity_name="robot",
  #   resampling_time_range=(20.0, 20.0),
  #   debug_vis=True,
  # )
  #
  # # Uniform-only: timer-based resampled uniform velocity ranges.
  # cfg.commands["twist"] = mdp.UniformVelocityCommandCfg(
  #   entity_name="robot",
  #   resampling_time_range=(3.0, 8.0),
  #   rel_standing_envs=0.1,
  #   rel_heading_envs=0.0,
  #   rel_forward_envs=0.0,
  #   heading_command=False,
  #   heading_control_stiffness=0.5,
  #   debug_vis=True,
  #   ranges=mdp.UniformVelocityCommandCfg.Ranges(
  #     lin_vel_x=(-2.0, 4.0),
  #     lin_vel_y=(-1.0, 1.0),
  #     ang_vel_z=(-0.8, 0.8),
  #   ),
  # )
  #
  # Mixed (spatial split): directional_fraction of envs run directional,
  # the rest run uniform — both command types operate simultaneously.
  cfg.commands["twist"] = mdp.MixedVelocityCommandCfg(
    directional_cfg=mdp.DirectionalVelocityCommandCfg(
      entity_name="robot",
      rel_forward=0.7, rel_backward=0.3,
      rel_left=0.0, rel_right=0.0,
      rel_standing=0.0,
      rel_ang_overlay=0.40,
      forward_speed=(0.5, 4.0),
      backward_speed=(0.5, 1.5),
      lateral_speed=(0.3, 0.8),
      ang_speed=(0.3, 0.8),
      debug_vis=False,
    ),
    uniform_cfg=mdp.UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.1,
      rel_heading_envs=0.0,
      rel_forward_envs=0.0,
      heading_command=False,
      heading_control_stiffness=0.5,
      debug_vis=False,
      ranges=mdp.UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.5, 4.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.8, 0.8),
      ),
    ),
    directional_fraction=0.5,  # 30% directional + 70% uniform
    debug_vis=True,
  )

  # --- Observations --------------------------------------------------------
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "twist"},
  )
  # phase_obs = ObservationTermCfg(
  #   func=mdp.phase,
  #   params={"period": 0.6, "command_name": "twist"},
  # )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs
  # cfg.observations["actor"].terms["phase"] = phase_obs
  # cfg.observations["critic"].terms["phase"] = phase_obs

  # --- Rewards -------------------------------------------------------------
  # task = velocity tracking, gated by SMP.
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "command_name": "twist",
      # "fixed_timesteps": (2, 5, 8, 15, 22),
      "task_terms": (
        (
            mdp.track_linear_velocity,
            1.0,
            {"command_name": "twist", "std": math.sqrt(1), "reverse_penalty": True},
        ),
        # (
        #   mdp.track_linear_velocity_world,
        #   1.0,
        #   {"command_name": "twist", "std": math.sqrt(2.0)},
        # ),
        (
            mdp.track_angular_velocity,
            0.5,
            {"command_name": "twist", "std": math.sqrt(1)},
        ),
        (
            mdp.action_rate_l2,
            -0.05,
            {},
        ),
        # (
        #   mdp.stand_still,
        #   -0.5,
        #   {
        #     "command_name": "twist",
        #     "command_threshold": 0.1,
        #     "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        #   },
        # ),
        # (
        #     mdp.stand_still,
        #     -0.3,
        #     {
        #         "command_name": "twist",
        #         "command_threshold": 0.1,
        #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        #     },
        # ),
        # (
        #   mdp.feet_gait,
        #   0.5,
        #   {
        #     "period": 0.6,
        #     "offset": [0.0, 0.5],
        #     "threshold": 0.56,
        #     "command_threshold": 0.1,
        #     "command_name": "twist",
        #     "sensor_name": "feet_ground_contact",
        #   },
        # ),
        # (
        #     mdp.feet_gait,
        #     0.3,
        #     {
        #         "period": 0.6,
        #         "offset": [0.0, 0.5],
        #         "threshold": 0.56,
        #         "command_threshold": 0.1,
        #         "command_name": "twist",
        #         "sensor_name": "feet_ground_contact",
        #     },
        # ),
      ),
    },
  )

  # # Zero-weight standalone tracking terms monitored by the curriculum.
  # cfg.rewards["track_lin_vel"] = RewardTermCfg(
  #   func=mdp.track_linear_velocity,
  #   weight=0.0,
  #   params={"command_name": "twist", "std": math.sqrt(0.5)},
  # )
  # cfg.rewards["track_ang_vel"] = RewardTermCfg(
  #   func=mdp.track_angular_velocity,
  #   weight=0.0,
  #   params={"command_name": "twist", "std": math.sqrt(0.5)},
  # )

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    # "datasets/pretrain_ckpt/pretrained_loco.pt"
    "logs/pretrain/pretrain/20260604_192628/pretrained.pt"
  )


  # --- Terminations --------------------------------------------------------
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  ##
  # Curriculum
  ##

  # cfg.curriculum = {
  #   # "terrain_levels": CurriculumTermCfg(
  #   #   func=mdp.terrain_levels_vel,
  #   #   params={"command_name": "twist"},
  #   # ),
  #   "velocity_cmd_levels": CurriculumTermCfg(
  #     func=mdp.velocity_cmd_levels,
  #     params={
  #       "command_name": "twist",
  #       "reward_term_names": ["task_smp_product"],
  #       "threshold_ratio": 0.8,
  #       "delta": 0.1,
  #     },
  #   ),
  # }

  return cfg
