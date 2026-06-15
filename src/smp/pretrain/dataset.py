"""Motion window dataset for diffusion pretraining."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from smp.motion.normalization import (
  denormalize_quantiles,
  fit_quantile_bounds,
  normalize_quantiles,
)


class MotionWindowDataset(Dataset[torch.Tensor]):
  """Loads pre-windowed NPZs produced by scripts/csv_to_npz.py.

  Normalization uses pre-computed q01/q99 quantiles (from
  ``scripts/compute_norm_stats.py``) to map features to [-1, 1].
  """

  def __init__(
    self,
    data_dir: str | Path,
    norm_stats_file: str | Path | None = None,
  ) -> None:
    npz_files = sorted(Path(data_dir).glob("*.npz"))
    if not npz_files:
      msg = f"No NPZ files found in {data_dir}"
      raise FileNotFoundError(msg)

    chunks: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for npz_file in npz_files:
      with np.load(npz_file, allow_pickle=False) as npz:
        windows = npz["windows"].astype(np.float32, copy=False)
      if windows.ndim != 3:
        msg = (
          f"{npz_file.name}: 'windows' has shape {windows.shape}, expected (N, W, S)"
        )
        raise ValueError(msg)
      if expected_shape is None:
        expected_shape = (int(windows.shape[1]), int(windows.shape[2]))
      elif (windows.shape[1], windows.shape[2]) != expected_shape:
        msg = (
          f"{npz_file.name}: shape {windows.shape} mismatches "
          f"first file's (*, {expected_shape[0]}, {expected_shape[1]})"
        )
        raise ValueError(msg)
      chunks.append(windows)

    assert expected_shape is not None
    self.window_size, self.feature_dim = expected_shape

    data = np.concatenate(chunks, axis=0)

    if norm_stats_file is not None:
      stats = np.load(norm_stats_file, allow_pickle=False)
      self.q_low = stats["q_low"].astype(np.float32)
      self.q_high = stats["q_high"].astype(np.float32)
    else:
      # Fallback: compute from data directly.
      flat = data.reshape(-1, self.feature_dim)
      self.q_low, self.q_high = fit_quantile_bounds(flat)

    # Normalize
    data = normalize_quantiles(data, self.q_low, self.q_high)

    self.windows = torch.from_numpy(data)

  def denormalize(self, x: torch.Tensor) -> torch.Tensor:
    q_low = torch.from_numpy(self.q_low).to(x.device, x.dtype)
    q_high = torch.from_numpy(self.q_high).to(x.device, x.dtype)
    return denormalize_quantiles(x, q_low, q_high)

  def __len__(self) -> int:
    return self.windows.shape[0]

  def __getitem__(self, idx: int) -> torch.Tensor:
    return self.windows[idx]
