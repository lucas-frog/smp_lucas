"""Pure-torch quaternion and rotation helpers for SMP motion code."""

from __future__ import annotations

import torch


def normalize_vector(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
  return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


def quat_pos(q: torch.Tensor) -> torch.Tensor:
  """Flip quaternions so the scalar component is non-negative."""
  mask = q[..., 0:1] < 0
  return torch.where(mask, -q, q)


def quat_unit(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
  return normalize_vector(q, eps=eps)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
  return torch.cat([q[..., 0:1], -q[..., 1:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Hamilton product for wxyz quaternions."""
  aw, ax, ay, az = a.unbind(dim=-1)
  bw, bx, by, bz = b.unbind(dim=-1)
  w = aw * bw - ax * bx - ay * by - az * bz
  x = aw * bx + ax * bw + ay * bz - az * by
  y = aw * by - ax * bz + ay * bw + az * bx
  z = aw * bz + ax * by - ay * bx + az * bw
  return torch.stack([w, x, y, z], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate vector(s) ``v`` by quaternion(s) ``q``."""
  q_w = q[..., 0:1]
  q_xyz = q[..., 1:]
  t = 2.0 * torch.cross(q_xyz, v, dim=-1)
  return v + q_w * t + torch.cross(q_xyz, t, dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  return quat_rotate(q, v)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  return quat_rotate(quat_conjugate(q), v)


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
  """Convert wxyz quaternion(s) to rotation matrix/matrices."""
  q = quat_pos(quat_unit(q))
  w, x, y, z = q.unbind(dim=-1)
  xx = x * x
  yy = y * y
  zz = z * z
  xy = x * y
  xz = x * z
  yz = y * z
  wx = w * x
  wy = w * y
  wz = w * z

  m00 = 1.0 - 2.0 * (yy + zz)
  m01 = 2.0 * (xy - wz)
  m02 = 2.0 * (xz + wy)
  m10 = 2.0 * (xy + wz)
  m11 = 1.0 - 2.0 * (xx + zz)
  m12 = 2.0 * (yz - wx)
  m20 = 2.0 * (xz - wy)
  m21 = 2.0 * (yz + wx)
  m22 = 1.0 - 2.0 * (xx + yy)

  return torch.stack(
    [m00, m01, m02, m10, m11, m12, m20, m21, m22], dim=-1
  ).reshape(q.shape[:-1] + (3, 3))


def quat_from_matrix(r: torch.Tensor) -> torch.Tensor:
  """Convert rotation matrix/matrices to wxyz quaternion(s)."""
  if r.shape[-2:] != (3, 3):
    msg = f"Expected (..., 3, 3) rotation matrices; got {tuple(r.shape)}"
    raise ValueError(msg)

  flat_r = r.reshape(-1, 3, 3)
  flat_q = torch.empty(flat_r.shape[0], 4, device=r.device, dtype=r.dtype)

  m00 = flat_r[:, 0, 0]
  m11 = flat_r[:, 1, 1]
  m22 = flat_r[:, 2, 2]
  trace = m00 + m11 + m22

  mask0 = trace > 0.0
  if mask0.any():
    s = torch.sqrt(trace[mask0] + 1.0) * 2.0
    flat_q[mask0, 0] = 0.25 * s
    flat_q[mask0, 1] = (flat_r[mask0, 2, 1] - flat_r[mask0, 1, 2]) / s
    flat_q[mask0, 2] = (flat_r[mask0, 0, 2] - flat_r[mask0, 2, 0]) / s
    flat_q[mask0, 3] = (flat_r[mask0, 1, 0] - flat_r[mask0, 0, 1]) / s

  mask1 = (~mask0) & (m00 >= m11) & (m00 >= m22)
  if mask1.any():
    s = torch.sqrt(1.0 + m00[mask1] - m11[mask1] - m22[mask1]) * 2.0
    flat_q[mask1, 0] = (flat_r[mask1, 2, 1] - flat_r[mask1, 1, 2]) / s
    flat_q[mask1, 1] = 0.25 * s
    flat_q[mask1, 2] = (flat_r[mask1, 0, 1] + flat_r[mask1, 1, 0]) / s
    flat_q[mask1, 3] = (flat_r[mask1, 0, 2] + flat_r[mask1, 2, 0]) / s

  mask2 = (~mask0) & (~mask1) & (m11 >= m22)
  if mask2.any():
    s = torch.sqrt(1.0 + m11[mask2] - m00[mask2] - m22[mask2]) * 2.0
    flat_q[mask2, 0] = (flat_r[mask2, 0, 2] - flat_r[mask2, 2, 0]) / s
    flat_q[mask2, 1] = (flat_r[mask2, 0, 1] + flat_r[mask2, 1, 0]) / s
    flat_q[mask2, 2] = 0.25 * s
    flat_q[mask2, 3] = (flat_r[mask2, 1, 2] + flat_r[mask2, 2, 1]) / s

  mask3 = (~mask0) & (~mask1) & (~mask2)
  if mask3.any():
    s = torch.sqrt(1.0 + m22[mask3] - m00[mask3] - m11[mask3]) * 2.0
    flat_q[mask3, 0] = (flat_r[mask3, 1, 0] - flat_r[mask3, 0, 1]) / s
    flat_q[mask3, 1] = (flat_r[mask3, 0, 2] + flat_r[mask3, 2, 0]) / s
    flat_q[mask3, 2] = (flat_r[mask3, 1, 2] + flat_r[mask3, 2, 1]) / s
    flat_q[mask3, 3] = 0.25 * s

  return quat_pos(quat_unit(flat_q.reshape(r.shape[:-2] + (4,))))


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
  """Extract the yaw-only quaternion from a full wxyz quaternion."""
  forward = torch.zeros_like(q[..., :3])
  forward[..., 0] = 1.0
  rotated = quat_rotate(q, forward)
  heading = torch.atan2(rotated[..., 1], rotated[..., 0])
  half = heading * 0.5
  yaw = torch.zeros_like(q)
  yaw[..., 0] = torch.cos(half)
  yaw[..., 3] = torch.sin(half)
  return quat_pos(yaw)


def tan_norm_from_quat(q: torch.Tensor) -> torch.Tensor:
  """Convert wxyz quaternion(s) to 6D tan-norm [col0, col2]."""
  mat = matrix_from_quat(q)
  return torch.cat([mat[..., :, 0], mat[..., :, 2]], dim=-1)


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
  """Convert 6D tan-norm [col0, col2] to rotation matrix/matrices."""
  col0 = d6[..., :3]
  col2 = d6[..., 3:6]
  col0 = normalize_vector(col0)
  col2 = col2 - (col0 * col2).sum(dim=-1, keepdim=True) * col0
  col2 = normalize_vector(col2)
  col1 = torch.cross(col2, col0, dim=-1)
  return torch.stack([col0, col1, col2], dim=-1)


def rot6d_to_quat(d6: torch.Tensor) -> torch.Tensor:
  return quat_from_matrix(rot6d_to_matrix(d6))
