"""Helpers for loading and sampling pretrained diffusion priors."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from smp.pretrain.feature_masks import build_upper_lower_feature_masks
from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler


class ClassifierFreeSampleModel(torch.nn.Module):
  """Wrap a conditioned model for classifier-free guidance at sampling time."""

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
    class_labels_dummy = torch.cat((class_labels, class_labels), dim=0)

    out = self.model(
      x_dummy,
      t_dummy,
      class_labels=class_labels_dummy,
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


def _denormalize_window(
  x_t: torch.Tensor,
  q_low: np.ndarray,
  q_high: np.ndarray,
  device: torch.device,
) -> torch.Tensor:
  q_low_t = torch.from_numpy(np.asarray(q_low, dtype=np.float32)).to(device)
  q_high_t = torch.from_numpy(np.asarray(q_high, dtype=np.float32)).to(device)
  return ((x_t.squeeze(0) + 1.0) / 2.0 * (q_high_t - q_low_t) + q_low_t).cpu()


def build_model_and_scheduler(
  ckpt: dict[str, Any],
  device: torch.device,
) -> tuple[DiffusionDenoiser, DDPMScheduler, np.ndarray, np.ndarray]:
  cfg = ckpt["cfg"]
  style_names = tuple(cfg.get("style_names", ()))
  model = DiffusionDenoiser(
    feature_dim=int(cfg["feature_dim"]),
    window_size=int(cfg["window_size"]),
    d_model=int(cfg.get("d_model", 256)),
    nhead=int(cfg.get("nhead", 8)),
    num_layers=int(cfg.get("num_layers", 2)),
    dropout=float(cfg.get("dropout", 0.0)),
    num_classes=len(style_names) if bool(cfg.get("conditional", False)) else 0,
    cfg_dropout=float(cfg.get("cfg_dropout", 0.0)),
  ).to(device)
  state = ckpt.get("model_ema") or ckpt["model"]
  model.load_state_dict(state)
  model.eval()

  scheduler = DDPMScheduler(
    num_timesteps=int(cfg.get("num_timesteps", 50)),
  ).to(device)
  return model, scheduler, ckpt["q_low"], ckpt["q_high"]


@torch.no_grad()
def sample_window(
  model: DiffusionDenoiser,
  scheduler: DDPMScheduler,
  q_low: np.ndarray,
  q_high: np.ndarray,
  window_size: int,
  feature_dim: int,
  device: torch.device,
  style_id: int | None = None,
  cfg_scale: float = 1.0,
  sampler: str = "ddpm",
  num_steps: int | None = None,
) -> torch.Tensor:
  x_t = torch.randn(1, window_size, feature_dim, device=device)
  class_labels = (
    torch.tensor([style_id], dtype=torch.long, device=device)
    if style_id is not None
    else None
  )
  if cfg_scale != 1.0 and class_labels is None:
    msg = "cfg_scale requires a conditioned style label"
    raise ValueError(msg)

  schedule = build_sampling_schedule(
    scheduler.num_timesteps,
    sampler=sampler,
    num_steps=num_steps,
  )
  for idx, t in enumerate(schedule):
    t_batch = torch.full((1,), t, dtype=torch.long, device=device)
    eps = _predict_eps(model, x_t, t_batch, class_labels=class_labels, cfg_scale=cfg_scale)
    if sampler == "ddpm":
      x_t = scheduler.step(eps, x_t, t)
    else:
      t_prev = schedule[idx + 1] if idx + 1 < len(schedule) else None
      x_t = _ddim_step(scheduler, eps, x_t, t, t_prev)

  return _denormalize_window(x_t, q_low, q_high, device)


@torch.no_grad()
def sample_window_composed(
  model: DiffusionDenoiser,
  scheduler: DDPMScheduler,
  q_low: np.ndarray,
  q_high: np.ndarray,
  window_size: int,
  feature_dim: int,
  device: torch.device,
  style_upper_id: int,
  style_lower_id: int,
  cfg_scale: float = 1.0,
  sampler: str = "ddim",
  num_steps: int | None = None,
) -> torch.Tensor:
  if cfg_scale != 1.0 and (style_upper_id is None or style_lower_id is None):
    msg = "cfg_scale requires conditioned style labels for composition"
    raise ValueError(msg)

  upper_mask, lower_mask = build_upper_lower_feature_masks(feature_dim)
  upper_mask = upper_mask.to(device=device).view(1, 1, feature_dim)
  lower_mask = lower_mask.to(device=device).view(1, 1, feature_dim)

  x_t = torch.randn(1, window_size, feature_dim, device=device)
  upper_labels = torch.tensor([style_upper_id], dtype=torch.long, device=device)
  lower_labels = torch.tensor([style_lower_id], dtype=torch.long, device=device)

  schedule = build_sampling_schedule(
    scheduler.num_timesteps,
    sampler=sampler,
    num_steps=num_steps,
  )
  for idx, t in enumerate(schedule):
    t_batch = torch.full((1,), t, dtype=torch.long, device=device)
    eps_upper = _predict_eps(
      model,
      x_t,
      t_batch,
      class_labels=upper_labels,
      cfg_scale=cfg_scale,
    )
    eps_lower = _predict_eps(
      model,
      x_t,
      t_batch,
      class_labels=lower_labels,
      cfg_scale=cfg_scale,
    )
    eps = upper_mask * eps_upper + lower_mask * eps_lower
    if sampler == "ddpm":
      x_t = scheduler.step(eps, x_t, t)
    else:
      t_prev = schedule[idx + 1] if idx + 1 < len(schedule) else None
      x_t = _ddim_step(scheduler, eps, x_t, t, t_prev)

  return _denormalize_window(x_t, q_low, q_high, device)
