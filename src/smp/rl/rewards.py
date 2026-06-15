"""Reward functions for SMP RL tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from smp.motion.normalization import normalize_quantiles
from smp.rl.utils import DiffNormalizer, MotionFeatureBuffer

if TYPE_CHECKING:
  from collections.abc import Callable

  from mjlab.envs import ManagerBasedRlEnv

  TaskTerm = tuple["Callable[..., torch.Tensor]", float, dict]


def _update_buffer_from_sim(env: ManagerBasedRlEnv) -> None:
  """Push current sim kinematics onto the buffer tail, env-origin-relative
  (matching ``_prime_sim_and_buffer``) so features are placement-invariant."""
  robot = env.scene["robot"]
  ee_indexes = env._smp_ee_indexes  # type: ignore[attr-defined]
  buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  origins = env.scene.env_origins
  buffer.update(
    robot.data.root_link_pos_w - origins,
    robot.data.root_link_quat_w,
    robot.data.root_link_lin_vel_w,
    robot.data.root_link_ang_vel_w,
    robot.data.body_link_pos_w[:, ee_indexes] - origins[:, None, :],
    robot.data.joint_pos,
    robot.data.joint_vel,
  )


def smp_guidance_reward(
  env: ManagerBasedRlEnv,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 4.0,
  normalize: bool = True,
  subsample_steps: int = 1,
) -> torch.Tensor:
  """SDS-style guidance reward over fixed timesteps ``K``:
  ``exp(-w_s/|K| · Σ_{i∈K} ‖ε̂_i − ε_i‖²)``.  ``normalize`` divides each MSE by a
  ``DiffNormalizer`` running mean (policy-relative) vs. raw (absolute scale);
  always stashes the mean raw MSE on ``env._smp_raw_err``.

  ``subsample_steps`` spaces buffer updates every N policy steps, extending the
  effective temporal span of the 10-frame window (e.g. ``subsample_steps=5`` with
  50 Hz policy → window covers 1.0 s instead of 0.2 s)."""
  device = torch.device(env.device)
  model, scheduler, q_low, q_high, _, _ = env._smp_bundle  # type: ignore[attr-defined]
  normalizer: DiffNormalizer = env._smp_normalizer  # type: ignore[attr-defined]
  buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]

  # Subsample buffer updates to widen the temporal window.
  if subsample_steps > 1:
    step = int(env.episode_length_buf[0].item())
    if step % subsample_steps == 0:
      _update_buffer_from_sim(env)
  else:
    _update_buffer_from_sim(env)

  features = buffer.compute_features()
  x_0 = normalize_quantiles(features, q_low, q_high, eps=1e-8)
  num_envs = x_0.shape[0]

  total_err = torch.zeros(num_envs, device=device)
  total_raw = torch.zeros(num_envs, device=device)
  with torch.no_grad():
    for t_scalar in fixed_timesteps:
      if not 0 <= t_scalar < scheduler.num_timesteps:
        msg = f"fixed_timestep {t_scalar} out of range [0, {scheduler.num_timesteps})"
        raise ValueError(msg)
      t = torch.full((num_envs,), t_scalar, dtype=torch.long, device=device)
      noise = torch.randn_like(x_0)
      x_t = scheduler.add_noise(x_0, noise, t)
      eps_hat = model(x_t, t)
      mse_per_env = ((eps_hat - noise) ** 2).mean(dim=(-1, -2))
      total_raw += mse_per_env
      if normalize:
        total_err += normalizer.update_and_normalize(t_scalar, mse_per_env)
      else:
        total_err += mse_per_env

  env._smp_raw_err = total_raw / len(fixed_timesteps)  # type: ignore[attr-defined]
  err = total_err / len(fixed_timesteps)
  return torch.exp(-err * ws)


def task_smp_product(
  env: ManagerBasedRlEnv,
  task_terms: tuple[TaskTerm, ...],
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
  command_name: str = "",
  subsample_steps: int = 1,
) -> torch.Tensor:
  """``(Σ wᵢ · taskᵢ(env)) · r_smp`` — multiplicative SMP gating; ``task_terms`` is
  a tuple of ``(func, weight, kwargs)``.  Calls ``smp_guidance_reward`` once (the
  sole SMP-buffer update), so it must be the task's only SMP reward term.

  When ``command_name`` is given and the corresponding command term carries an
  ``is_standing_env`` mask, environments where it is ``True`` skip the SMP gate
  (``r_smp = 1.0``) so the motion prior does not penalise static standing poses
  that are under-represented in the locomotion dataset.

  ``subsample_steps`` is forwarded to ``smp_guidance_reward`` to control the
  effective temporal window of the motion-prior feature buffer.
  """
  r_smp = smp_guidance_reward(
    env,
    fixed_timesteps=fixed_timesteps,
    ws=ws,
    subsample_steps=subsample_steps,
  )
  # if command_name:
  #   cmd_term = env.command_manager.get_term(command_name)
  #   if cmd_term is not None and hasattr(cmd_term, "is_standing_env"):
  #     standing = cmd_term.is_standing_env
  #     r_smp = torch.where(standing, torch.ones_like(r_smp), r_smp)
  task = sum(w * func(env, **kw) for func, w, kw in task_terms)
  return task * r_smp
