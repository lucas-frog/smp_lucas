from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest import TestCase, mock

import torch

from smp.rl import rewards
from smp.rl.tasks.localization import mdp as localization_mdp
from smp.rl.tasks.localization import velocity_env_cfg
from mjlab.managers.scene_entity_config import SceneEntityCfg


def _term_linear(_env: object) -> torch.Tensor:
  return torch.tensor([1.0, 2.0])


def _term_angular(_env: object) -> torch.Tensor:
  return torch.tensor([0.5, 1.5])


class RewardSplitLoggingTests(TestCase):
  def test_combined_reward_modes_share_single_step_cache(self) -> None:
    env = SimpleNamespace(device="cpu", common_step_counter=7)
    task_terms = (
      (_term_linear, 1.0, {}),
      (_term_angular, -0.5, {}),
    )
    metric_task_terms = copy.deepcopy(task_terms)

    calls: list[tuple[tuple[int, ...], float, int]] = []

    def _fake_smp(
      _env: object,
      fixed_timesteps: tuple[int, ...] = (8, 15, 22),
      ws: float = 4.0,
      normalize: bool = True,
      subsample_steps: int = 1,
    ) -> torch.Tensor:
      del normalize
      calls.append((fixed_timesteps, ws, subsample_steps))
      _env._smp_raw_err = torch.tensor([3.0, 4.0])
      return torch.tensor([0.25, 0.75])

    with mock.patch("smp.rl.rewards.smp_guidance_reward", side_effect=_fake_smp):
      product_reward = rewards.combined_reward(
        env,
        task_terms=task_terms,
        fixed_timesteps=(1, 4),
        ws=2.0,
        subsample_steps=3,
        combine_mode="product",
        task_scale=9.0,
        smp_scale=11.0,
      )
      task_reward = rewards.task_reward_metric(
        env,
        task_terms=metric_task_terms,
        fixed_timesteps=(1, 4),
        ws=2.0,
        subsample_steps=3,
      )
      smp_reward = rewards.smp_reward_metric(
        env,
        task_terms=metric_task_terms,
        fixed_timesteps=(1, 4),
        ws=2.0,
        subsample_steps=3,
      )
      product_metric = rewards.total_reward_metric(
        env,
        task_terms=metric_task_terms,
        fixed_timesteps=(1, 4),
        ws=2.0,
        subsample_steps=3,
        combine_mode="product",
        task_scale=9.0,
        smp_scale=11.0,
      )
      linear_metric = rewards.task_term_metric(
        env,
        term_name="_term_linear",
        task_terms=metric_task_terms,
        fixed_timesteps=(1, 4),
        ws=2.0,
        subsample_steps=3,
      )
      raw_err_metric = rewards.smp_raw_err_metric(
        env,
        task_terms=metric_task_terms,
        fixed_timesteps=(1, 4),
        ws=2.0,
        subsample_steps=3,
      )

    self.assertEqual(calls, [((1, 4), 2.0, 3)])
    self.assertTrue(torch.equal(task_reward, torch.tensor([0.75, 1.25])))
    self.assertTrue(torch.equal(smp_reward, torch.tensor([0.25, 0.75])))
    self.assertTrue(torch.equal(product_reward, torch.tensor([0.1875, 0.9375])))
    self.assertTrue(torch.equal(product_metric, torch.tensor([0.1875, 0.9375])))
    self.assertTrue(torch.equal(linear_metric, torch.tensor([1.0, 2.0])))
    self.assertTrue(torch.equal(raw_err_metric, torch.tensor([3.0, 4.0])))

    env.common_step_counter = 8
    with mock.patch("smp.rl.rewards.smp_guidance_reward", side_effect=_fake_smp):
      sum_reward = rewards.combined_reward(
        env,
        task_terms=task_terms,
        fixed_timesteps=(1, 4),
        ws=2.0,
        subsample_steps=3,
        combine_mode="sum",
        task_scale=2.0,
        smp_scale=3.0,
      )

    self.assertEqual(calls, [((1, 4), 2.0, 3), ((1, 4), 2.0, 3)])
    self.assertTrue(torch.equal(sum_reward, torch.tensor([2.25, 4.75])))

  def test_velocity_env_can_switch_between_product_and_sum(self) -> None:
    product_cfg = velocity_env_cfg.g1_velocity_smp_env_cfg(play=False)
    sum_cfg = velocity_env_cfg.g1_velocity_smp_env_cfg(
      play=False,
      reward_mode="sum",
      task_scale=1.5,
      smp_scale=0.25,
    )

    self.assertIn("total_reward", product_cfg.rewards)
    self.assertNotIn("task_smp_product", product_cfg.rewards)
    self.assertEqual(
      product_cfg.rewards["total_reward"].params["combine_mode"], "product"
    )
    self.assertEqual(sum_cfg.rewards["total_reward"].params["combine_mode"], "sum")
    self.assertEqual(sum_cfg.rewards["total_reward"].params["task_scale"], 1.5)
    self.assertEqual(sum_cfg.rewards["total_reward"].params["smp_scale"], 0.25)

    self.assertIn("task_reward", product_cfg.metrics)
    self.assertIn("smp_reward", product_cfg.metrics)
    self.assertIn("total_reward", product_cfg.metrics)
    self.assertIn("task_track_linear_velocity", product_cfg.metrics)
    self.assertIn("task_track_angular_velocity", product_cfg.metrics)
    self.assertIn("task_action_rate_l2", product_cfg.metrics)
    self.assertIn("smp_raw_err", product_cfg.metrics)

  def test_velocity_env_includes_amp_inspired_regularizers(self) -> None:
    cfg = velocity_env_cfg.g1_velocity_smp_env_cfg(play=False)

    task_terms = cfg.rewards["total_reward"].params["task_terms"]
    task_term_names = [func.__name__ for func, _weight, _params in task_terms]

    self.assertIn("joint_pos_limits", task_term_names)
    self.assertIn("body_ang_vel", task_term_names)
    self.assertIn("task_joint_pos_limits", cfg.metrics)
    self.assertIn("task_body_ang_vel", cfg.metrics)

  def test_body_ang_vel_returns_scalar_per_env_for_multiple_bodies(self) -> None:
    robot = SimpleNamespace(
      data=SimpleNamespace(
        body_link_ang_vel_w=torch.tensor(
          [
            [[1.0, 2.0, 3.0], [0.5, 0.0, 1.0], [2.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.5, 0.5, 0.0], [0.0, 0.0, 2.0]],
          ]
        )
      )
    )
    env = SimpleNamespace(scene={"robot": robot})

    reward = localization_mdp.body_ang_vel(
      env,
      asset_cfg=SceneEntityCfg("robot", body_ids=[0, 1, 2]),
    )

    self.assertEqual(reward.shape, torch.Size([2]))
    self.assertTrue(torch.equal(reward, torch.tensor([10.25, 3.5])))

  def test_body_ang_vel_resolves_body_names_inside_task_terms(self) -> None:
    robot = SimpleNamespace(
      data=SimpleNamespace(
        body_link_ang_vel_w=torch.tensor(
          [
            [[1.0, 2.0, 3.0], [0.5, 0.0, 1.0], [2.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.5, 0.5, 0.0], [0.0, 0.0, 2.0]],
          ]
        )
      ),
      find_bodies=mock.Mock(return_value=([1], ["torso_link"])),
    )
    env = SimpleNamespace(scene={"robot": robot})

    reward = localization_mdp.body_ang_vel(
      env,
      asset_cfg=SceneEntityCfg("robot", body_names=("torso_link",)),
    )

    robot.find_bodies.assert_called_once_with(("torso_link",), preserve_order=False)
    self.assertTrue(torch.equal(reward, torch.tensor([0.25, 2.5])))
