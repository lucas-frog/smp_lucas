"""Feature-level masks for style composition on G1 motion windows."""

from __future__ import annotations

import torch

FEATURE_DIM_G1 = 59


def build_upper_lower_feature_masks(feature_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
  """Return binary ``(upper_mask, lower_mask)`` for the 59-dim G1 feature layout.

  Layout:
    root_pos(3), root_rot(6), joint_pos(29), ee_pos(15), root_lin_vel(3), root_ang_vel(3)

  Joint split:
    lower = legs
    upper = waist + both arms

  EE split:
    lower = left/right foot
    upper = torso + both wrists
  """
  if feature_dim != FEATURE_DIM_G1:
    msg = f"Only feature_dim={FEATURE_DIM_G1} is supported, got {feature_dim}"
    raise ValueError(msg)

  upper_mask = torch.zeros(feature_dim, dtype=torch.float32)
  lower_mask = torch.zeros(feature_dim, dtype=torch.float32)

  # root_pos + root_rot
  lower_mask[0:9] = 1.0

  # joint_pos
  lower_mask[9:21] = 1.0
  upper_mask[21:38] = 1.0

  # ee_pos: [left_foot, right_foot, torso, left_wrist, right_wrist]
  lower_mask[38:44] = 1.0
  upper_mask[44:53] = 1.0

  # root_lin_vel + root_ang_vel
  lower_mask[53:59] = 1.0

  return upper_mask, lower_mask
