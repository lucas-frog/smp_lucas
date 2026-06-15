"""Shared motion feature and normalization utilities."""

from smp.motion.features import (
  EE_BODY_NAMES,
  FEATURE_DIM,
  FEATURE_DIMS,
  NUM_EE,
  NUM_JOINTS,
  compute_motion_features,
  slice_motion_features,
)
from smp.motion.math import tan_norm_from_quat
from smp.motion.normalization import (
  denormalize_quantiles,
  fit_quantile_bounds,
  normalize_quantiles,
)

__all__ = [
  "EE_BODY_NAMES",
  "FEATURE_DIM",
  "FEATURE_DIMS",
  "NUM_EE",
  "NUM_JOINTS",
  "compute_motion_features",
  "denormalize_quantiles",
  "fit_quantile_bounds",
  "normalize_quantiles",
  "slice_motion_features",
  "tan_norm_from_quat",
]
