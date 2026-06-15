"""q01/q99 quantile normalization helpers for SMP motion features."""

from __future__ import annotations

import numpy as np
import torch


def fit_quantile_bounds(
  frames: np.ndarray,
  q_low: float = 0.01,
  q_high: float = 0.99,
  min_span: float = 1e-6,
  fallback_span: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
  """Compute per-feature quantile bounds with a fallback for constant features."""
  low = np.percentile(frames, q_low * 100, axis=0).astype(np.float32)
  high = np.percentile(frames, q_high * 100, axis=0).astype(np.float32)
  tiny = (high - low) < min_span
  if tiny.any():
    high[tiny] = low[tiny] + fallback_span
  return low, high


def normalize_quantiles(
  x: torch.Tensor | np.ndarray,
  q_low: torch.Tensor | np.ndarray,
  q_high: torch.Tensor | np.ndarray,
  eps: float = 0.0,
) -> torch.Tensor | np.ndarray:
  """Map raw features to the diffusion model's q01/q99 normalized space."""
  return 2.0 * (x - q_low) / (q_high - q_low + eps) - 1.0


def denormalize_quantiles(
  x: torch.Tensor | np.ndarray,
  q_low: torch.Tensor | np.ndarray,
  q_high: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
  """Map q01/q99 normalized features back to raw feature space."""
  return (x + 1.0) / 2.0 * (q_high - q_low) + q_low
