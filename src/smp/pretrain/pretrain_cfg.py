"""Diffusion pretraining configuration."""

from __future__ import annotations

from dataclasses import dataclass

from smp.utils import detect_device


@dataclass
class PretrainCfg:
  """Configuration for diffusion model pretraining."""

  # Data
  data_dir: str = "datasets/npz"
  norm_stats_file: str = "datasets/norm_stats.npz"
  """Path to q01/q99 quantile stats from compute_norm_stats.py."""
  train_split: float = 0.9

  # Model. ``d_model = nhead · head_dim`` is the DiT inner dim; FF inner
  # dim is fixed at 4·d_model.
  arch_name: str = "DiT"
  """Denoiser architecture: 'DiT' or MimicKit-compatible 'CondDiT'/'cdit'."""
  d_model: int = 256
  nhead: int = 4
  num_layers: int = 2
  dropout: float = 0.0
  conditional: bool = False
  """Enable class-conditioned DiT when the dataset provides style labels."""
  cfg_dropout: float = 0.0
  """Classifier-free guidance label dropout used during conditional training."""

  # Diffusion
  num_timesteps: int = 50
  num_noise_samples: int = 10
  """Random (t, ε) draws per data point in the diffusion loss."""

  # EMA
  use_ema: bool = False
  ema_decay: float = 0.9999

  # Training
  batch_size: int = 1024
  num_epochs: int = 2000
  lr: float = 3e-4
  weight_decay: float = 1e-4
  max_grad_norm: float = 1.0

  # Logging
  name: str = "pretrain"
  """Run identifier; used as the run name and the save subfolder."""
  log_interval: int = 10
  save_interval: int = 100
  log_dir: str = "logs/pretrain"
  logger: str = "wandb"
  """Logging backend: 'wandb', 'tensorboard', or 'none'."""
  wandb_project: str = "smp"
  """W&B project name (only used when logger='wandb')."""

  # Device
  device: str = ""

  # Reproducibility
  seed: int = 42

  def __post_init__(self) -> None:
    if not self.device:
      self.device = detect_device()
