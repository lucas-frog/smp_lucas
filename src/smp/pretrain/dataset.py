"""Motion window dataset for diffusion pretraining."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MotionWindowDataset(
  Dataset[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]
):
  """Loads pre-windowed NPZs produced by scripts/csv_to_npz.py.

  Supports two layouts:

  - Unconditional: ``data_dir/*.npz``
  - Class-conditioned: ``data_dir/<style_name>/*.npz``

  Normalization uses pre-computed q01/q99 quantiles (from
  ``scripts/compute_norm_stats.py``) to map features to [-1, 1].
  """

  def __init__(
    self,
    data_dir: str | Path,
    norm_stats_file: str | Path | None = None,
  ) -> None:
    root = Path(data_dir)
    norm_stats_path = Path(norm_stats_file).resolve() if norm_stats_file is not None else None

    flat_npz_files = self._filtered_npz_files(root.glob("*.npz"), norm_stats_path)
    style_dirs: list[tuple[str, list[Path]]] = []
    for child in sorted(root.iterdir()):
      if not child.is_dir():
        continue
      style_npz = self._filtered_npz_files(child.glob("*.npz"), norm_stats_path)
      if style_npz:
        style_dirs.append((child.name, style_npz))

    if style_dirs and flat_npz_files:
      msg = (
        f"{root} mixes flat .npz files with style subdirectories. "
        "Use either unconditional flat files or conditioned style subdirectories."
      )
      raise ValueError(msg)

    self.style_names: tuple[str, ...] = ()
    self.num_classes = 0
    self.is_conditioned = False

    chunks: list[np.ndarray] = []
    style_id_chunks: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None

    if style_dirs:
      self.style_names = tuple(name for name, _files in style_dirs)
      self.num_classes = len(self.style_names)
      self.is_conditioned = True
      for style_id, (_style_name, npz_files) in enumerate(style_dirs):
        for npz_file in npz_files:
          windows = self._load_windows(npz_file, expected_shape)
          expected_shape = (int(windows.shape[1]), int(windows.shape[2]))
          chunks.append(windows)
          style_id_chunks.append(
            np.full((windows.shape[0],), style_id, dtype=np.int64)
          )
    else:
      if not flat_npz_files:
        msg = f"No NPZ files found in {data_dir}"
        raise FileNotFoundError(msg)
      for npz_file in flat_npz_files:
        windows = self._load_windows(npz_file, expected_shape)
        expected_shape = (int(windows.shape[1]), int(windows.shape[2]))
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
      self.q_low = np.percentile(flat, 1, axis=0).astype(np.float32)
      self.q_high = np.percentile(flat, 99, axis=0).astype(np.float32)
      span = self.q_high - self.q_low
      tiny = span < 1e-6
      if tiny.any():
        self.q_high[tiny] = self.q_low[tiny] + 1.0

    data = 2.0 * (data - self.q_low) / (self.q_high - self.q_low) - 1.0
    self.windows = torch.from_numpy(data)
    self.style_ids = (
      torch.from_numpy(np.concatenate(style_id_chunks, axis=0))
      if self.is_conditioned
      else None
    )

  @staticmethod
  def _filtered_npz_files(
    files: list[Path] | object,
    norm_stats_path: Path | None,
  ) -> list[Path]:
    filtered: list[Path] = []
    for file_path in sorted(files):
      if norm_stats_path is not None and file_path.resolve() == norm_stats_path:
        continue
      filtered.append(file_path)
    return filtered

  @staticmethod
  def _load_windows(
    npz_file: Path,
    expected_shape: tuple[int, int] | None,
  ) -> np.ndarray:
    with np.load(npz_file, allow_pickle=False) as npz:
      windows = npz["windows"].astype(np.float32, copy=False)
    if windows.ndim != 3:
      msg = (
        f"{npz_file.name}: 'windows' has shape {windows.shape}, expected (N, W, S)"
      )
      raise ValueError(msg)
    if expected_shape is not None and (windows.shape[1], windows.shape[2]) != expected_shape:
      msg = (
        f"{npz_file.name}: shape {windows.shape} mismatches "
        f"first file's (*, {expected_shape[0]}, {expected_shape[1]})"
      )
      raise ValueError(msg)
    return windows

  def denormalize(self, x: torch.Tensor) -> torch.Tensor:
    q_low = torch.from_numpy(self.q_low).to(x.device, x.dtype)
    q_high = torch.from_numpy(self.q_high).to(x.device, x.dtype)
    return (x + 1.0) / 2.0 * (q_high - q_low) + q_low

  def __len__(self) -> int:
    return self.windows.shape[0]

  def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if self.style_ids is None:
      return self.windows[idx]
    return self.windows[idx], self.style_ids[idx]
