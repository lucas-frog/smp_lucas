"""G1 recovery dual-task config.

Half of the environments track a full-body reference trajectory from an npz
file. The other half keep the localization velocity objective with SMP guidance.
The actor uses only the shared base observations; the critic additionally sees a
task id from the recovery command wrapper.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise  # noqa: F401

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.tasks.localization import mdp

G1_TRACKING_BODY_NAMES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)

G1_END_EFFECTOR_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)
DEFAULT_RECOVERY_TRAJECTORY_FILE = "/home/lucas/origin/unitree_rl_mjlab/src/assets/motions/g1/fall.npz"
# DEFAULT_RECOVERY_TRAJECTORY_FILE = "/home/lucas/dataset/tracking/walk.npz"

VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}


def _tracking_body_asset_cfg() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot",
    body_names=G1_TRACKING_BODY_NAMES,
    preserve_order=True,
  )


def _velocity_command_cfg() -> mdp.MixedVelocityCommandCfg:
  return mdp.MixedVelocityCommandCfg(
    directional_cfg=mdp.DirectionalVelocityCommandCfg(
      entity_name="robot",
      rel_forward=0.7,
      rel_backward=0.3,
      rel_left=0.0,
      rel_right=0.0,
      rel_standing=0.0,
      rel_ang_overlay=0.40,
      forward_speed=(0.5, 3.0),
      backward_speed=(0.5, 1.0),
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
    directional_fraction=1.0,
    debug_vis=True,
  )


def g1_recovery_smp_env_cfg(
  play: bool = False,
  *,
  trajectory_motion_file: str = DEFAULT_RECOVERY_TRAJECTORY_FILE,
) -> ManagerBasedRlEnvCfg:
  """Build the G1 recovery env cfg with trajectory + velocity sub-tasks."""
  cfg = g1_smp_env_cfg(play=play)

  # --- Commands ------------------------------------------------------------
  # Previous velocity-layout wrapper kept for rollback:
  # cfg.commands["recovery_task"] = mdp.RecoveryTaskCommandCfg(...)
  cfg.commands["recovery_task"] = mdp.RecoveryGoalCommandCfg(
    trajectory_cfg=mdp.RecoveryTrajectoryCommandCfg(
      motion_file=trajectory_motion_file,
      entity_name="robot",
      anchor_body_name="torso_link",
      body_names=G1_TRACKING_BODY_NAMES,
      end_effector_body_names=G1_END_EFFECTOR_BODY_NAMES,
      sampling_mode="start" if play else "adaptive",
      pose_range={}
      if play
      else {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range={} if play else VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      future_velocity_window_steps=50,
    ),
    velocity_cfg=_velocity_command_cfg(),
    trajectory_fraction=0.5,
    goal_horizon_s=1.0,
    standing_height=0.65,
    stand_up_threshold=0.65,
    stand_up_speed_range=(0.2, 0.8),
    debug_vis=True,
  )

  # --- Observations --------------------------------------------------------
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "recovery_task"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs

  cfg.observations["critic"].terms["task_id"] = ObservationTermCfg(
    func=mdp.recovery_task_id,
    params={"command_name": "recovery_task"},
  )
  cfg.observations["critic"].terms["root_height"] = ObservationTermCfg(
    func=mdp.robot_root_height,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  # cfg.observations["actor"].terms["motion_anchor_pos_b"] = ObservationTermCfg(
  #   func=mdp.motion_anchor_pos_b,
  #   params={"command_name": "recovery_task"},
  #   noise=Unoise(n_min=-0.25, n_max=0.25),
  # )
  # cfg.observations["actor"].terms["motion_anchor_ori_b"] = ObservationTermCfg(
  #   func=mdp.motion_anchor_ori_b,
  #   params={"command_name": "recovery_task"},
  #   noise=Unoise(n_min=-0.05, n_max=0.05),
  # )
  # cfg.observations["critic"].terms["motion_anchor_pos_b"] = ObservationTermCfg(
  #   func=mdp.motion_anchor_pos_b,
  #   params={"command_name": "recovery_task"},
  # )
  # cfg.observations["critic"].terms["motion_anchor_ori_b"] = ObservationTermCfg(
  #   func=mdp.motion_anchor_ori_b,
  #   params={"command_name": "recovery_task"},
  # )
  cfg.observations["critic"].terms["body_pos"] = ObservationTermCfg(
    func=mdp.robot_body_pos_b,
    params={"command_name": "recovery_task", "asset_cfg": _tracking_body_asset_cfg()},
  )
  cfg.observations["critic"].terms["body_ori"] = ObservationTermCfg(
    func=mdp.robot_body_ori_b,
    params={"command_name": "recovery_task", "asset_cfg": _tracking_body_asset_cfg()},
  )
  cfg.observations["critic"].terms["body_lin_vel"] = ObservationTermCfg(
    func=mdp.robot_body_lin_vel_b,
    params={"command_name": "recovery_task", "asset_cfg": _tracking_body_asset_cfg()},
  )
  cfg.observations["critic"].terms["body_ang_vel"] = ObservationTermCfg(
    func=mdp.robot_body_ang_vel_b,
    params={"command_name": "recovery_task", "asset_cfg": _tracking_body_asset_cfg()},
  )

  # --- Rewards -------------------------------------------------------------
  # cfg.rewards["motion_global_root_pos"] = RewardTermCfg(
  #   func=mdp.motion_global_anchor_position_error_exp,
  #   weight=0.5,
  #   params={"command_name": "recovery_task", "std": 0.3},
  # )
  # cfg.rewards["motion_global_root_ori"] = RewardTermCfg(
  #   func=mdp.motion_global_anchor_orientation_error_exp,
  #   weight=0.5,
  #   params={"command_name": "recovery_task", "std": 0.4},
  # )
  cfg.rewards["trajectory_tracking"] = RewardTermCfg(
    func=mdp.trajectory_tracking_reward,
    weight=1.0,
    params={
      "command_name": "recovery_task",
      "tracking_cfg": mdp.TrajectoryTrackingRewardCfg(),
    },
  )
  # Previous separate Gaussian tracking rewards kept for rollback:
  # cfg.rewards["motion_root_height"] = RewardTermCfg(
  #   func=mdp.motion_root_height_error_exp,
  #   weight=0.5,
  #   params={"command_name": "recovery_task", "std": 0.3},
  # )
  # cfg.rewards["motion_body_pos"] = RewardTermCfg(
    # func=mdp.motion_relative_body_position_error_exp,
  #   weight=1.0,
  #   params={"command_name": "recovery_task", "std": 0.3},
  # )
  # cfg.rewards["motion_body_ori"] = RewardTermCfg(
  #   func=mdp.motion_relative_body_orientation_error_exp,
  #   weight=1.0,
  #   params={"command_name": "recovery_task", "std": 0.4},
  # )
  # cfg.rewards["motion_body_lin_vel"] = RewardTermCfg(
  #   func=mdp.motion_global_body_linear_velocity_error_exp,
  #   weight=1.0,
  #   params={"command_name": "recovery_task", "std": 1.0},
  # )
  # cfg.rewards["motion_body_ang_vel"] = RewardTermCfg(
  #   func=mdp.motion_global_body_angular_velocity_error_exp,
  #   weight=1.0,
  #   params={"command_name": "recovery_task", "std": 3.14},
  # )
  cfg.rewards["velocity_smp_product"] = RewardTermCfg(
    func=mdp.masked_velocity_smp_reward,
    weight=5.0,
    params={
      "command_name": "recovery_task",
      "task_terms": (
        # (
        #   mdp.track_linear_velocity_upright,
        #   1.0,
        #   {
        #     "command_name": "recovery_task",
        #     "std": math.sqrt(1.0),
        #     "reverse_penalty": True,
        #     "root_height_threshold": 0.65,
        #     "root_height_ramp_start": 0.55,
        #   },
        # ),
        # (
        #   mdp.track_angular_velocity_upright,
        #   0.5,
        #   {
        #     "command_name": "recovery_task",
        #     "std": math.sqrt(1.0),
        #     "root_height_threshold": 0.65,
        #     "root_height_ramp_start": 0.55,
        #   },
        # ),
        (
          mdp.track_goal_root_position,
          1.0,
          {
            "command_name": "recovery_task",
            "std": math.sqrt(1.0),
            "horizon_s": 1.0,
          },
        ),
        (
          mdp.track_goal_root_orientation,
          0.5,
          {
            "command_name": "recovery_task",
            "std": math.sqrt(1.0),
            "horizon_s": 1.0,
          },
        ),
        (mdp.action_rate_l2, -0.05, {}),
        (
            mdp.joint_pos_limits,
            -10.0,
            {},
        ),
        (
          mdp.root_height_penalty,
          1.0,
          {"target_height": 0.65, "std": 0.2},
        ),
      ),
    },
  )
  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=mdp.tracking_only_action_rate_l2,
    weight=-0.05,
    params={"command_name": "recovery_task"},
  )
  cfg.rewards["joint_limit"] = RewardTermCfg(
    func=mdp.tracking_only_joint_pos_limits,
    weight=-10.0,
    params={
      "command_name": "recovery_task",
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.tracking_only_self_collision_cost,
    weight=-10.0,
    params={
      "command_name": "recovery_task",
      "sensor_name": "self_collision",
      "force_threshold": 10.0,
    },
  )

  # --- Terminations ---------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "/home/lucas/dataset/pretrain_model/walk_getup/pretrained.pt"
    # "/home/lucas/dataset/pretrain_model/walkrun/pretrained.pt"
  )

  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.velocity_only_root_height_below_minimum,
    params={
      "command_name": "recovery_task",
      "minimum_height": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
      "grace_steps": 120,
    },
  )
  cfg.terminations["self_collision"] = TerminationTermCfg(
    func=mdp.velocity_only_illegal_contact,
    params={"command_name": "recovery_task", "sensor_name": "self_collision"},
  )
  cfg.terminations["tracking_anchor_pos_z"] = TerminationTermCfg(
    func=mdp.bad_anchor_pos_z_only,
    params={
      "command_name": "recovery_task",
      "threshold": 0.25,
      "grace_steps": 0,
    },
  )
  cfg.terminations["tracking_anchor_ori"] = TerminationTermCfg(
    func=mdp.bad_anchor_ori,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "command_name": "recovery_task",
      "threshold": 0.8,
    },
  )
  cfg.terminations["tracking_body_pos_z"] = TerminationTermCfg(
    func=mdp.bad_motion_body_pos_z_only,
    params={
      "command_name": "recovery_task",
      "threshold": 0.5,
      "body_names": G1_END_EFFECTOR_BODY_NAMES,
      "grace_steps": 0,
    },
  )

  return cfg


__all__ = ["g1_recovery_smp_env_cfg"]
