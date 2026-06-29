from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import importlib
import sys

import numpy as np
import torch

from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler
from smp.rl import env_cfg, events, rewards, utils


class _StyleAwareModel(torch.nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.calls: list[tuple[int, int]] = []

  def forward(
    self,
    x_t: torch.Tensor,
    t: torch.Tensor,
    class_labels: torch.Tensor | None = None,
    force_drop_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    del force_drop_ids
    label = -1 if class_labels is None else int(class_labels[0].item())
    self.calls.append((label, int(t[0].item())))
    return torch.full_like(x_t, float(label))


class _ReusedBufferModel(torch.nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self._buffer: torch.Tensor | None = None

  def forward(
    self,
    x_t: torch.Tensor,
    t: torch.Tensor,
    class_labels: torch.Tensor | None = None,
    force_drop_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    del t, force_drop_ids
    label = -1.0 if class_labels is None else float(class_labels[0].item())
    if self._buffer is None or self._buffer.shape != x_t.shape or self._buffer.device != x_t.device:
      self._buffer = torch.empty_like(x_t)
    self._buffer.fill_(label)
    return self._buffer


class _ZeroModel(torch.nn.Module):
  def forward(
    self,
    x_t: torch.Tensor,
    t: torch.Tensor,
    class_labels: torch.Tensor | None = None,
    force_drop_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    del t, class_labels, force_drop_ids
    return torch.zeros_like(x_t)


class _RecordingNormalizer:
  def __init__(self) -> None:
    self.calls: list[int] = []

  def update_and_normalize(self, t: int, mse_per_env: torch.Tensor) -> torch.Tensor:
    self.calls.append(t)
    return mse_per_env


class _StaticBuffer:
  def __init__(self, features: torch.Tensor) -> None:
    self._features = features

  def compute_features(self) -> torch.Tensor:
    return self._features


def _make_conditional_ckpt(path: Path) -> None:
  model = DiffusionDenoiser(
    feature_dim=59,
    window_size=10,
    d_model=32,
    nhead=4,
    num_layers=1,
    dropout=0.0,
    num_classes=2,
    cfg_dropout=0.1,
  )
  ckpt = {
    "cfg": {
      "feature_dim": 59,
      "window_size": 10,
      "d_model": 32,
      "nhead": 4,
      "num_layers": 1,
      "dropout": 0.0,
      "num_timesteps": 6,
      "conditional": True,
      "cfg_dropout": 0.1,
      "style_names": ("walk", "run"),
    },
    "model": model.state_dict(),
    "q_low": np.zeros(59, dtype=np.float32),
    "q_high": np.ones(59, dtype=np.float32),
  }
  torch.save(ckpt, path)


class ComposedPriorRlTests(unittest.TestCase):
  def test_load_denoiser_preserves_single_style_compatibility(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt_path = Path(tmpdir) / "pretrained.pt"
      model = DiffusionDenoiser(
        feature_dim=8,
        window_size=4,
        d_model=32,
        nhead=4,
        num_layers=1,
      )
      ckpt = {
        "cfg": {
          "feature_dim": 8,
          "window_size": 4,
          "d_model": 32,
          "nhead": 4,
          "num_layers": 1,
          "num_timesteps": 5,
        },
        "model": model.state_dict(),
        "q_low": np.zeros(8, dtype=np.float32),
        "q_high": np.ones(8, dtype=np.float32),
      }
      torch.save(ckpt, ckpt_path)

      bundle = utils.load_denoiser(str(ckpt_path), "cpu")

      self.assertEqual(bundle["feature_dim"], 8)
      self.assertEqual(bundle["window_size"], 4)
      self.assertFalse(bundle["conditional"])
      self.assertEqual(bundle["style_names"], ())
      self.assertEqual(bundle["prior_mode"], "single")

  def test_load_denoiser_restores_conditional_metadata(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt_path = Path(tmpdir) / "pretrained.pt"
      _make_conditional_ckpt(ckpt_path)

      bundle = utils.load_denoiser(str(ckpt_path), "cpu")

      self.assertTrue(bundle["conditional"])
      self.assertEqual(bundle["style_names"], ("walk", "run"))
      self.assertEqual(bundle["style_name_to_id"]["walk"], 0)
      self.assertEqual(bundle["style_name_to_id"]["run"], 1)
      self.assertEqual(bundle["model"].num_classes, 2)
      self.assertAlmostEqual(bundle["model"].cfg_dropout, 0.1)

  def test_predict_prior_noise_single_style_matches_old_path(self) -> None:
    model = _ZeroModel()
    x_t = torch.randn(2, 3, 4)
    t = torch.tensor([1, 1], dtype=torch.long)
    bundle = {
      "model": model,
      "feature_dim": 4,
      "conditional": False,
      "prior_mode": "single",
    }

    eps = utils.predict_prior_noise(bundle, x_t, t)

    self.assertTrue(torch.equal(eps, torch.zeros_like(x_t)))

  def test_predict_prior_noise_composed_uses_upper_and_lower_styles(self) -> None:
    model = _StyleAwareModel()
    x_t = torch.randn(1, 2, 59)
    t = torch.tensor([4], dtype=torch.long)
    bundle = {
      "model": model,
      "feature_dim": 59,
      "conditional": True,
      "prior_mode": "composed",
      "style_upper_id": 0,
      "style_lower_id": 1,
      "cfg_scale": 1.0,
    }

    eps = utils.predict_prior_noise(bundle, x_t, t)

    self.assertEqual(model.calls, [(0, 4), (1, 4)])
    self.assertTrue(torch.all(eps[..., 21:38] == 0.0))
    self.assertTrue(torch.all(eps[..., 44:53] == 0.0))
    self.assertTrue(torch.all(eps[..., 0:9] == 1.0))
    self.assertTrue(torch.all(eps[..., 9:21] == 1.0))
    self.assertTrue(torch.all(eps[..., 38:44] == 1.0))
    self.assertTrue(torch.all(eps[..., 53:59] == 1.0))

  def test_predict_prior_noise_composed_clones_reused_outputs(self) -> None:
    model = _ReusedBufferModel()
    x_t = torch.randn(1, 2, 59)
    t = torch.tensor([4], dtype=torch.long)
    bundle = {
      "model": model,
      "feature_dim": 59,
      "conditional": True,
      "prior_mode": "composed",
      "style_upper_id": 0,
      "style_lower_id": 1,
      "cfg_scale": 1.0,
    }

    eps = utils.predict_prior_noise(bundle, x_t, t)

    self.assertTrue(torch.all(eps[..., 21:38] == 0.0))
    self.assertTrue(torch.all(eps[..., 44:53] == 0.0))
    self.assertTrue(torch.all(eps[..., 0:9] == 1.0))
    self.assertTrue(torch.all(eps[..., 9:21] == 1.0))
    self.assertTrue(torch.all(eps[..., 38:44] == 1.0))
    self.assertTrue(torch.all(eps[..., 53:59] == 1.0))

  def test_sample_prior_windows_routes_to_composed_ddim(self) -> None:
    model = _StyleAwareModel()
    scheduler = DDPMScheduler(num_timesteps=6)
    bundle = {
      "model": model,
      "scheduler": scheduler,
      "q_low": torch.zeros(59),
      "q_high": torch.ones(59),
      "feature_dim": 59,
      "window_size": 3,
      "conditional": True,
      "prior_mode": "composed",
      "style_upper_id": 0,
      "style_lower_id": 1,
      "cfg_scale": 1.0,
      "sampler": "ddim",
      "num_steps": 3,
    }

    windows = utils.sample_prior_windows(bundle, n=1, device=torch.device("cpu"))

    self.assertEqual(tuple(windows.shape), (1, 3, 59))
    self.assertEqual(
      model.calls,
      [(0, 5), (1, 5), (0, 2), (1, 2), (0, 0), (1, 0)],
    )

  def test_smp_guidance_reward_uses_composed_prior_when_configured(self) -> None:
    features = torch.zeros(2, 3, 59)
    env = SimpleNamespace()
    env.device = "cpu"
    env.scene = {"robot": object()}
    env._smp_bundle = {
      "model": _StyleAwareModel(),
      "scheduler": DDPMScheduler(num_timesteps=6),
      "q_low": torch.zeros(59),
      "q_high": torch.ones(59),
      "feature_dim": 59,
      "window_size": 3,
      "conditional": True,
      "prior_mode": "composed",
      "style_upper_id": 0,
      "style_lower_id": 1,
      "cfg_scale": 1.0,
    }
    env._smp_normalizer = _RecordingNormalizer()
    env._smp_buffer = _StaticBuffer(features)
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)

    with mock.patch("smp.rl.rewards._update_buffer_from_sim") as update_buffer:
      reward = rewards.smp_guidance_reward(
        env,
        fixed_timesteps=(1, 4),
        ws=1.0,
        normalize=True,
      )

    self.assertEqual(tuple(reward.shape), (2,))
    self.assertEqual(env._smp_bundle["model"].calls, [(0, 1), (1, 1), (0, 4), (1, 4)])
    self.assertEqual(env._smp_normalizer.calls, [1, 4])
    update_buffer.assert_called_once()

  def test_init_smp_state_passes_prior_config_into_sampling_bundle(self) -> None:
    env = SimpleNamespace()
    env.device = torch.device("cpu")
    env.num_envs = 2
    env.common_step_counter = 0
    env.cfg = SimpleNamespace(sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=0.005)), decimation=4)
    env.scene = {
      "robot": SimpleNamespace(
        find_bodies=lambda names, preserve_order=True: ([0, 1, 2, 3, 4], None),
        data=SimpleNamespace(default_root_state=torch.zeros(2, 13)),
      ),
    }

    fake_bundle = {
      "model": _ZeroModel(),
      "scheduler": DDPMScheduler(num_timesteps=5),
      "q_low": torch.zeros(59),
      "q_high": torch.ones(59),
      "feature_dim": 59,
      "window_size": 10,
      "conditional": True,
      "style_names": ("walk", "run"),
      "style_name_to_id": {"walk": 0, "run": 1},
      "prior_mode": "single",
    }

    with (
      mock.patch("smp.rl.events.load_denoiser", return_value=fake_bundle),
      mock.patch("smp.rl.events._maybe_compile", side_effect=lambda model, compile_model, compile_mode: model),
      mock.patch("smp.rl.events._sample_windows", side_effect=lambda env, n: torch.zeros(n, 10, 59)),
      mock.patch("smp.rl.events.gsi_reset") as gsi_reset,
    ):
      events.init_smp_state(
        env,
        ckpt_path="/tmp/fake.pt",
        gsi_buffer_size=4,
        gsi_batch_size=2,
        compile_model=False,
        prior_mode="composed",
        style_upper="walk",
        style_lower="run",
        cfg_scale=1.5,
        sampler="ddim",
        num_steps=4,
      )

    self.assertEqual(env._smp_bundle["prior_mode"], "composed")
    self.assertEqual(env._smp_bundle["style_upper_id"], 0)
    self.assertEqual(env._smp_bundle["style_lower_id"], 1)
    self.assertEqual(env._smp_bundle["cfg_scale"], 1.5)
    self.assertEqual(env._smp_bundle["sampler"], "ddim")
    self.assertEqual(env._smp_bundle["num_steps"], 4)
    self.assertEqual(tuple(env._smp_gsi_pool.shape), (4, 10, 59))
    gsi_reset.assert_called_once()

  def test_configure_smp_prior_updates_only_init_params(self) -> None:
    cfg = SimpleNamespace(
      events={
        "init_smp_state": SimpleNamespace(params={"ckpt_path": "old.pt", "compile_model": True}),
        "gsi_refresh": SimpleNamespace(params={"num_samples": 1024}),
      }
    )

    env_cfg.configure_smp_prior(
      cfg,
      ckpt_path="new.pt",
      prior_mode="composed",
      style_upper="walk",
      style_lower="run",
      cfg_scale=1.25,
      sampler="ddim",
      num_steps=20,
    )

    params = cfg.events["init_smp_state"].params
    self.assertEqual(params["ckpt_path"], "new.pt")
    self.assertEqual(params["prior_mode"], "composed")
    self.assertEqual(params["style_upper"], "walk")
    self.assertEqual(params["style_lower"], "run")
    self.assertEqual(params["cfg_scale"], 1.25)
    self.assertEqual(params["sampler"], "ddim")
    self.assertEqual(params["num_steps"], 20)
    self.assertEqual(cfg.events["gsi_refresh"].params["num_samples"], 1024)

  def test_velocity_composed_env_cfg_sets_composed_prior_without_touching_default(self) -> None:
    import smp.rl.tasks.localization.velocity_env_cfg as velocity_env_cfg

    default_cfg = velocity_env_cfg.g1_velocity_smp_env_cfg(play=False)
    composed_cfg = velocity_env_cfg.g1_velocity_composed_smp_env_cfg(play=False)

    default_params = default_cfg.events["init_smp_state"].params
    composed_params = composed_cfg.events["init_smp_state"].params

    self.assertNotIn("prior_mode", default_params)
    self.assertEqual(composed_params["prior_mode"], "composed")
    self.assertEqual(
      composed_params["ckpt_path"],
      "/home/hjqsyy/smp/logs/pretrain/cond_walk4_ws10_v1/20260622_213521/pretrained.pt",
    )
    self.assertEqual(composed_params["style_upper"], "walk_guai1")
    self.assertEqual(composed_params["style_lower"], "walk_guai2")
    self.assertEqual(composed_params["cfg_scale"], 1.0)
    self.assertEqual(composed_params["sampler"], "ddim")
    self.assertEqual(composed_params["num_steps"], 10)

  def test_velocity_single_style_env_cfg_sets_single_prior_from_multistyle_ckpt(self) -> None:
    import smp.rl.tasks.localization.velocity_env_cfg as velocity_env_cfg

    single_style_cfg = velocity_env_cfg.g1_velocity_single_style_smp_env_cfg(play=False)
    params = single_style_cfg.events["init_smp_state"].params

    self.assertEqual(params["prior_mode"], "single")
    self.assertEqual(
      params["ckpt_path"],
      "/home/hjqsyy/smp/logs/pretrain/cond_walk4_ws10_v1/20260622_213521/pretrained.pt",
    )
    self.assertEqual(params["style"], "walk_guai2")
    self.assertEqual(params["cfg_scale"], 1.0)
    self.assertEqual(params["sampler"], "ddim")
    self.assertEqual(params["num_steps"], 10)
    self.assertNotIn("style_upper", params)
    self.assertNotIn("style_lower", params)

  def test_velocity_walkrun_product_env_cfg_uses_legacy_product_reward_stack(self) -> None:
    import smp.rl.tasks.localization.velocity_env_cfg as velocity_env_cfg

    walkrun_cfg = velocity_env_cfg.g1_velocity_walkrun_product_smp_env_cfg(play=False)
    params = walkrun_cfg.events["init_smp_state"].params
    task_terms = walkrun_cfg.rewards["total_reward"].params["task_terms"]
    task_term_names = [func.__name__ for func, _weight, _params in task_terms]

    self.assertEqual(
      task_term_names,
      [
        "track_linear_velocity",
        "track_angular_velocity",
        "action_rate_l2",
      ],
    )
    self.assertEqual(walkrun_cfg.rewards["total_reward"].params["combine_mode"], "product")
    self.assertEqual(
      params["ckpt_path"],
      "/home/hjqsyy/smp/logs/pretrain/cond_walkrun_getup1_ws10_balanced_v1/20260628_181948/checkpoint_02900.pt",
    )
    self.assertEqual(params["prior_mode"], "single")
    self.assertEqual(params["style"], "walkrun")
    self.assertEqual(params["cfg_scale"], 1.0)
    self.assertEqual(params["sampler"], "ddim")
    self.assertEqual(params["num_steps"], 10)
    self.assertNotIn("task_joint_pos_limits", walkrun_cfg.metrics)
    self.assertNotIn("task_body_ang_vel", walkrun_cfg.metrics)
    self.assertNotIn("task_variable_posture", walkrun_cfg.metrics)

  def test_localization_registers_composed_task(self) -> None:
    module_name = "smp.rl.tasks.localization"
    sys.modules.pop(module_name, None)

    with mock.patch("mjlab.tasks.registry.register_mjlab_task") as register_task:
      importlib.import_module(module_name)

    task_ids = [call.kwargs["task_id"] for call in register_task.call_args_list]
    self.assertIn("Smp-Velocity-G1", task_ids)
    self.assertIn("Smp-Velocity-Sum-G1", task_ids)
    self.assertIn("Smp-Velocity-Composed-G1", task_ids)
    self.assertIn("Smp-Velocity-SingleStyle-G1", task_ids)
    self.assertIn("Smp-Velocity-SingleStyle-Walkrun-Product-G1", task_ids)


if __name__ == "__main__":
  unittest.main()
