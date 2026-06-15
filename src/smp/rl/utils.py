"""Utilities for SMP RL: denoiser loader, diff-normalizer, and feature buffer."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

from smp.motion.features import compute_motion_features
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
) -> tuple[DiffusionDenoiser, DDPMScheduler, torch.Tensor, torch.Tensor, int, int]:
  """Load a frozen pretrained denoiser checkpoint → ``(model, scheduler, q_low,
  q_high, feature_dim, window_size)``.

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

  model = DiffusionDenoiser(
    feature_dim=feature_dim,
    window_size=window_size,
    d_model=int(cfg.get("d_model", 256)),
    nhead=int(cfg.get("nhead", 8)),
    num_layers=int(cfg.get("num_layers", 2)),
    dropout=float(cfg.get("dropout", 0.0)),
  ).to(device)
  model.load_state_dict(state, strict=False)
  model.eval()
  model.requires_grad_(False)

  scheduler = DDPMScheduler(
    num_timesteps=int(cfg.get("num_timesteps", 50)),
  ).to(device)

  q_low = torch.from_numpy(np.asarray(q_low, dtype=np.float32)).to(device)
  q_high = torch.from_numpy(np.asarray(q_high, dtype=np.float32)).to(device)

  return model, scheduler, q_low, q_high, feature_dim, window_size


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
