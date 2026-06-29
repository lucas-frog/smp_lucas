"""Utilities for SMP RL: denoiser loader, diff-normalizer, and feature buffer."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

from smp.motion.features import compute_motion_features
from smp.motion.normalization import denormalize_quantiles
from smp.pretrain.feature_masks import build_upper_lower_feature_masks
from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler


def _remap_tinymdm_to_denoiser(state_dict: dict[str, Any]) -> dict[str, Any]:
  """Remap ``TinyStableMotionDiTModel`` / ``CondTinyStableMotionDiTModel``
  state-dict keys to ``DiffusionDenoiser`` keys so training checkpoints
  from ``unitree_rl_mjlab`` can be loaded directly."""
  remapped: dict[str, Any] = {}
  for key, value in state_dict.items():
    # ── optional prefix (dmodel., ema_model., etc.) ──
    new_key = key
    # Strip known outer prefixes — they are re-added by the caller
    for prefix in ("dmodel.", "ema_model.", "model."):
      if new_key.startswith(prefix):
        new_key = new_key[len(prefix):]
        break

    # transformer_blocks.N.attn1.  →  blocks.N.
    new_key = re.sub(
      r"^transformer_blocks\.(\d+)\.attn1\.", r"blocks.\1.", new_key
    )
    # transformer_blocks.N. (non-attn keys like ff, scale_shift_table) → blocks.N.
    new_key = re.sub(
      r"^transformer_blocks\.(\d+)\.", r"blocks.\1.", new_key
    )

    # ff.net.0.proj  →  ff.act.proj
    new_key = new_key.replace("ff.net.0.proj", "ff.act.proj")
    # ff.net.2  →  ff.proj_out
    new_key = new_key.replace("ff.net.2", "ff.proj_out")

    # adaln_single.emb.  →  adaln_single.  (drop ".emb" sub-module)
    new_key = new_key.replace("adaln_single.emb.", "adaln_single.")

    # to_out.0.bias — DiffusionDenoiser uses bias=False → skip
    if "to_out.0.bias" in new_key:
      continue
    # to_out.0.weight  →  to_out.weight
    new_key = new_key.replace("to_out.0.", "to_out.")

    # Top-level scale_shift_table (unused in forward) → skip
    if new_key == "scale_shift_table":
      continue

    remapped[new_key] = value
  return remapped


def _is_tinymdm_checkpoint(ckpt: dict[str, Any]) -> bool:
  """Heuristic: training checkpoints have flat ``dmodel.*`` keys and no ``cfg``."""
  return (
    "cfg" not in ckpt
    and any(k.startswith("dmodel.") for k in ckpt)
  )


def load_denoiser(
  ckpt_path: str,
  device: torch.device | str,
) -> dict[str, Any]:
  """Load a frozen pretrained denoiser checkpoint and return a metadata bundle.

  Accepts two checkpoint formats:

  * **SMP native** — ``{"model": state, "model_ema": state, "cfg": {...},
    "q_low": ndarray, "q_high": ndarray}``
  * **TinyMDM training** — flat ``dmodel.*`` / ``ema_dmodel.*`` keys,
    ``_smp_q_low`` / ``_smp_q_high``, no ``cfg``.  Module names are remapped
    to the ``DiffusionDenoiser`` layout.
  """
  device = torch.device(device)

  ckpt: dict[str, Any] = torch.load(ckpt_path, map_location=device, weights_only=False)

  # ── detect TinyMDM training format ──
  if _is_tinymdm_checkpoint(ckpt):
    # --- cfg from heuristics + known config defaults ---
    pe = ckpt["dmodel.sequence_pos_encoder.pe"]
    window_size = int(pe.shape[1])
    d_model = int(pe.shape[2])
    feature_dim = int(ckpt["dmodel.preprocess_conv.weight"].shape[0])
    num_layers = sum(
      1
      for k in ckpt
      if k.startswith("dmodel.transformer_blocks.")
      and ".attn1.to_q.weight" in k
    )

    cfg: dict[str, Any] = {
      "feature_dim": feature_dim,
      "window_size": window_size,
      "d_model": d_model,
      "nhead": 4,
      "num_layers": num_layers,
      "dropout": 0.05,
      "num_timesteps": 50,
    }

    # --- model weights ---
    raw_model = {k: v for k, v in ckpt.items() if not k.startswith("ema_dmodel.")}
    state = _remap_tinymdm_to_denoiser(raw_model)

    # --- EMA weights (preferred) ---
    ema_raw = {
      k: v
      for k, v in ckpt.items()
      if k.startswith("ema_dmodel.ema_model.") or k.startswith("ema_dmodel.model.")
    }
    if ema_raw:
      state = _remap_tinymdm_to_denoiser(ema_raw)

    # --- quantile bounds ---
    q_low = ckpt.get("_smp_q_low", ckpt.get("q_low"))
    q_high = ckpt.get("_smp_q_high", ckpt.get("q_high"))
  else:
    cfg = ckpt["cfg"]
    state = ckpt.get("model_ema") or ckpt["model"]
    # Remap if the native-format checkpoint still has TinyMDM key names
    # (can happen when a training checkpoint was partially converted).
    if any("transformer_blocks." in k or "adaln_single.emb." in k for k in state):
      state = _remap_tinymdm_to_denoiser(state)
    q_low = ckpt["q_low"]
    q_high = ckpt["q_high"]

  feature_dim = int(cfg["feature_dim"])
  window_size = int(cfg["window_size"])
  conditional = bool(cfg.get("conditional", False))
  style_names = tuple(cfg.get("style_names", ()))

  model = DiffusionDenoiser(
    feature_dim=feature_dim,
    window_size=window_size,
    d_model=int(cfg.get("d_model", 256)),
    nhead=int(cfg.get("nhead", 8)),
    num_layers=int(cfg.get("num_layers", 2)),
    dropout=float(cfg.get("dropout", 0.0)),
    num_classes=len(style_names) if conditional else 0,
    cfg_dropout=float(cfg.get("cfg_dropout", 0.0)) if conditional else 0.0,
  ).to(device)
  model.load_state_dict(state, strict=False)
  model.eval()
  model.requires_grad_(False)

  scheduler = DDPMScheduler(
    num_timesteps=int(cfg.get("num_timesteps", 50)),
  ).to(device)

  q_low = torch.from_numpy(np.asarray(q_low, dtype=np.float32)).to(device)
  q_high = torch.from_numpy(np.asarray(q_high, dtype=np.float32)).to(device)

  return {
    "model": model,
    "scheduler": scheduler,
    "q_low": q_low,
    "q_high": q_high,
    "feature_dim": feature_dim,
    "window_size": window_size,
    "conditional": conditional,
    "cfg_dropout": float(cfg.get("cfg_dropout", 0.0)) if conditional else 0.0,
    "style_names": style_names,
    "style_name_to_id": {name: idx for idx, name in enumerate(style_names)},
    "prior_mode": "single",
    "style_id": None,
    "style_upper_id": None,
    "style_lower_id": None,
    "cfg_scale": 1.0,
    "sampler": "ddpm",
    "num_steps": None,
  }


class ClassifierFreeSampleModel(torch.nn.Module):
  """Wrap a conditioned model for classifier-free guidance at inference time."""

  def __init__(self, model: DiffusionDenoiser) -> None:
    super().__init__()
    self.model = model

  def forward(
    self,
    x_t: torch.Tensor,
    t: torch.Tensor,
    class_labels: torch.Tensor,
    cfg_scale: float = 1.0,
  ) -> torch.Tensor:
    batch_size = x_t.shape[0]
    force_drop_ids = torch.cat(
      (
        torch.zeros(batch_size, device=x_t.device, dtype=torch.bool),
        torch.ones(batch_size, device=x_t.device, dtype=torch.bool),
      ),
      dim=0,
    )
    x_dummy = torch.cat((x_t, x_t), dim=0)
    t_dummy = torch.cat((t, t), dim=0)
    labels_dummy = torch.cat((class_labels, class_labels), dim=0)
    out = self.model(
      x_dummy,
      t_dummy,
      class_labels=labels_dummy,
      force_drop_ids=force_drop_ids,
    )
    cond_pred, uncond_pred = out.chunk(2, dim=0)
    return uncond_pred + cfg_scale * (cond_pred - uncond_pred)


def resolve_style_label(style_names: tuple[str, ...], style: str) -> int:
  if style not in style_names:
    msg = f"Unknown style '{style}'. Available styles: {', '.join(style_names)}"
    raise ValueError(msg)
  return style_names.index(style)


def build_sampling_schedule(
  num_timesteps: int,
  sampler: str = "ddpm",
  num_steps: int | None = None,
) -> list[int]:
  sampler = sampler.lower()
  if sampler == "ddpm":
    if num_steps is not None:
      msg = "num_steps is only supported with sampler='ddim'"
      raise ValueError(msg)
    return list(range(num_timesteps - 1, -1, -1))
  if sampler != "ddim":
    msg = f"Unknown sampler '{sampler}'. Expected 'ddpm' or 'ddim'."
    raise ValueError(msg)
  if num_steps is None:
    msg = "num_steps must be provided when sampler='ddim'"
    raise ValueError(msg)
  if not 1 <= num_steps <= num_timesteps:
    msg = f"num_steps must be in [1, {num_timesteps}], got {num_steps}"
    raise ValueError(msg)

  timesteps = np.linspace(num_timesteps - 1, 0, num_steps)
  schedule = [int(round(t)) for t in timesteps.tolist()]
  for idx in range(1, len(schedule)):
    if schedule[idx] >= schedule[idx - 1]:
      schedule[idx] = max(schedule[idx - 1] - 1, 0)
  schedule[-1] = 0
  return schedule


def _ddim_step(
  scheduler: DDPMScheduler,
  eps: torch.Tensor,
  x_t: torch.Tensor,
  t: int,
  t_prev: int | None,
) -> torch.Tensor:
  alpha_t = scheduler.alphas_cumprod[t]
  sqrt_alpha_t = torch.sqrt(alpha_t)
  sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
  x_0_hat = (x_t - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t
  if t_prev is None:
    return x_0_hat
  alpha_prev = scheduler.alphas_cumprod[t_prev]
  return torch.sqrt(alpha_prev) * x_0_hat + torch.sqrt(1.0 - alpha_prev) * eps


def _predict_eps(
  model: DiffusionDenoiser,
  x_t: torch.Tensor,
  t_batch: torch.Tensor,
  class_labels: torch.Tensor | None = None,
  cfg_scale: float = 1.0,
) -> torch.Tensor:
  if class_labels is None:
    return model(x_t, t_batch)
  if cfg_scale == 1.0:
    return model(x_t, t_batch, class_labels=class_labels)
  return ClassifierFreeSampleModel(model)(
    x_t,
    t_batch,
    class_labels=class_labels,
    cfg_scale=cfg_scale,
  )


def predict_prior_noise(
  bundle: dict[str, Any],
  x_t: torch.Tensor,
  t_batch: torch.Tensor,
) -> torch.Tensor:
  """Predict prior noise for either single-style or composed-style conditioning."""
  model: DiffusionDenoiser = bundle["model"]
  cfg_scale = float(bundle.get("cfg_scale", 1.0))
  if bundle.get("prior_mode", "single") != "composed":
    style_id = bundle.get("style_id")
    class_labels = (
      torch.full(
        (x_t.shape[0],),
        int(style_id),
        dtype=torch.long,
        device=x_t.device,
      )
      if style_id is not None
      else None
    )
    return _predict_eps(model, x_t, t_batch, class_labels=class_labels, cfg_scale=cfg_scale)

  feature_dim = int(bundle["feature_dim"])
  upper_mask, lower_mask = build_upper_lower_feature_masks(feature_dim)
  upper_mask = upper_mask.to(device=x_t.device).view(1, 1, feature_dim)
  lower_mask = lower_mask.to(device=x_t.device).view(1, 1, feature_dim)
  upper_id = int(bundle["style_upper_id"])
  lower_id = int(bundle["style_lower_id"])
  upper_labels = torch.full((x_t.shape[0],), upper_id, dtype=torch.long, device=x_t.device)
  lower_labels = torch.full((x_t.shape[0],), lower_id, dtype=torch.long, device=x_t.device)
  eps_upper = _predict_eps(
    model,
    x_t,
    t_batch,
    class_labels=upper_labels,
    cfg_scale=cfg_scale,
  ).clone()
  eps_lower = _predict_eps(
    model,
    x_t,
    t_batch,
    class_labels=lower_labels,
    cfg_scale=cfg_scale,
  ).clone()
  return upper_mask * eps_upper + lower_mask * eps_lower


@torch.no_grad()
def sample_prior_windows(
  bundle: dict[str, Any],
  n: int,
  device: torch.device,
) -> torch.Tensor:
  """Sample denormalized motion windows from the configured prior bundle."""
  model: DiffusionDenoiser = bundle["model"]
  scheduler: DDPMScheduler = bundle["scheduler"]
  q_low: torch.Tensor = bundle["q_low"]
  q_high: torch.Tensor = bundle["q_high"]
  feature_dim = int(bundle["feature_dim"])
  window_size = int(bundle["window_size"])
  sampler = str(bundle.get("sampler", "ddpm")).lower()
  num_steps = bundle.get("num_steps")

  x_t = torch.randn(n, window_size, feature_dim, device=device)
  schedule = build_sampling_schedule(
    scheduler.num_timesteps,
    sampler=sampler,
    num_steps=num_steps,
  )
  for idx, t in enumerate(schedule):
    t_batch = torch.full((n,), t, dtype=torch.long, device=device)
    eps = predict_prior_noise(bundle, x_t, t_batch)
    if sampler == "ddpm":
      x_t = scheduler.step(eps, x_t, t)
    else:
      t_prev = schedule[idx + 1] if idx + 1 < len(schedule) else None
      x_t = _ddim_step(scheduler, eps, x_t, t, t_prev)
  return denormalize_quantiles(x_t, q_low, q_high)


class DiffNormalizer:
  """Count-based running mean per diffusion timestep: equal-weighted, so it
  freezes as the count grows — a stable SDS-MSE reference, unlike a drifting EMA."""

  def __init__(
    self,
    num_timesteps: int,
    device: torch.device,
    min_value: float = 1e-4,
    max_count: int = 50_000_000,
  ) -> None:
    self.min_value = min_value
    self.max_count = max_count
    self.mean = torch.ones(num_timesteps, device=device)
    self.count = torch.zeros(num_timesteps, device=device, dtype=torch.long)

  def update_and_normalize(self, t: int, mse_per_env: torch.Tensor) -> torch.Tensor:
    """Record MSE values for timestep ``t``; return them divided by the mean."""
    if self.count[t] > self.max_count:
      # Freeze once stable (and avoid count overflow).
      return mse_per_env / self.mean[t].clamp(min=self.min_value)
    n = mse_per_env.numel()
    batch_mean = mse_per_env.mean()
    old_count = self.count[t].item()
    new_count = old_count + n
    if old_count == 0:
      self.mean[t] = batch_mean
    else:
      w_old = old_count / new_count
      w_new = n / new_count
      self.mean[t] = w_old * self.mean[t] + w_new * batch_mean
    self.count[t] = new_count
    return mse_per_env / self.mean[t].clamp(min=self.min_value)


class MotionFeatureBuffer:
  """Rolling per-env buffer of the last ``window_size`` kinematic samples;
  ``compute_features()`` returns a window anchored at the LAST frame's yaw-only
  local frame, layout (matching ``scripts/csv_to_npz.py``):

      ``[root_pos(3), root_rot(6), joint_pos(J), ee_pos(E*3),
         root_lin_vel(3), root_ang_vel(3)]``

  ``joint_vel`` is stored for symmetry but excluded from the output.  Positions
  use the caller's frame (SMP RL feeds env-origin-relative)."""

  def __init__(
    self,
    num_envs: int,
    window_size: int,
    num_joints: int,
    num_ee: int,
    device: torch.device | str,
  ) -> None:
    self.num_envs = num_envs
    self.window_size = window_size
    self.num_joints = num_joints
    self.num_ee = num_ee
    self.device = torch.device(device)

    self.root_pos_w = torch.zeros(num_envs, window_size, 3, device=self.device)
    self.root_quat_w = torch.zeros(num_envs, window_size, 4, device=self.device)
    self.root_quat_w[..., 0] = 1.0
    self.root_lin_vel_w = torch.zeros(num_envs, window_size, 3, device=self.device)
    self.root_ang_vel_w = torch.zeros(num_envs, window_size, 3, device=self.device)
    self.ee_pos_w = torch.zeros(num_envs, window_size, num_ee, 3, device=self.device)
    self.joint_pos = torch.zeros(num_envs, window_size, num_joints, device=self.device)
    self.joint_vel = torch.zeros(num_envs, window_size, num_joints, device=self.device)

  def reset(
    self,
    env_ids: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    ee_pos_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Fill all W slots of ``env_ids`` with a pre-sampled trajectory."""
    if env_ids.numel() == 0:
      return
    self.root_pos_w[env_ids] = root_pos_w
    self.root_quat_w[env_ids] = root_quat_w
    self.root_lin_vel_w[env_ids] = root_lin_vel_w
    self.root_ang_vel_w[env_ids] = root_ang_vel_w
    self.ee_pos_w[env_ids] = ee_pos_w
    self.joint_pos[env_ids] = joint_pos
    self.joint_vel[env_ids] = joint_vel

  def update(
    self,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    ee_pos_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Shift left by one and append the new frame at index W-1."""
    self.root_pos_w = torch.roll(self.root_pos_w, shifts=-1, dims=1)
    self.root_quat_w = torch.roll(self.root_quat_w, shifts=-1, dims=1)
    self.root_lin_vel_w = torch.roll(self.root_lin_vel_w, shifts=-1, dims=1)
    self.root_ang_vel_w = torch.roll(self.root_ang_vel_w, shifts=-1, dims=1)
    self.ee_pos_w = torch.roll(self.ee_pos_w, shifts=-1, dims=1)
    self.joint_pos = torch.roll(self.joint_pos, shifts=-1, dims=1)
    self.joint_vel = torch.roll(self.joint_vel, shifts=-1, dims=1)
    self.root_pos_w[:, -1] = root_pos_w
    self.root_quat_w[:, -1] = root_quat_w
    self.root_lin_vel_w[:, -1] = root_lin_vel_w
    self.root_ang_vel_w[:, -1] = root_ang_vel_w
    self.ee_pos_w[:, -1] = ee_pos_w
    self.joint_pos[:, -1] = joint_pos
    self.joint_vel[:, -1] = joint_vel

  def compute_features(self) -> torch.Tensor:
    """Return features ``(num_envs, W, FEATURE_DIM)`` in the shared SMP layout."""
    return compute_motion_features(
      root_pos=self.root_pos_w,
      root_quat=self.root_quat_w,
      root_lin_vel=self.root_lin_vel_w,
      root_ang_vel=self.root_ang_vel_w,
      ee_pos=self.ee_pos_w,
      joint_pos=self.joint_pos,
    )
