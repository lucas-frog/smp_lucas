"""Reward functions for SMP RL tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from smp.motion.normalization import normalize_quantiles
from smp.rl.utils import DiffNormalizer, MotionFeatureBuffer, predict_prior_noise

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
  bundle = env._smp_bundle  # type: ignore[attr-defined]
  scheduler = bundle["scheduler"]
  q_low = bundle["q_low"]
  q_high = bundle["q_high"]
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
      eps_hat = predict_prior_noise(bundle, x_t, t)
      mse_per_env = ((eps_hat - noise) ** 2).mean(dim=(-1, -2))
      total_raw += mse_per_env
      if normalize:
        total_err += normalizer.update_and_normalize(t_scalar, mse_per_env)
      else:
        total_err += mse_per_env

  env._smp_raw_err = total_raw / len(fixed_timesteps)  # type: ignore[attr-defined]
  err = total_err / len(fixed_timesteps)
  return torch.exp(-err * ws)


def _task_terms_signature(task_terms: tuple[TaskTerm, ...]) -> tuple:
  """Build a stable signature for task-term caching across deep-copied cfgs."""
  return tuple(
    (
      func.__module__,
      func.__qualname__,
      float(weight),
      tuple(sorted((key, repr(value)) for key, value in kwargs.items())),
    )
    for func, weight, kwargs in task_terms
  )


def _ensure_task_smp_cache(
  env: ManagerBasedRlEnv,
  task_terms: tuple[TaskTerm, ...],
  fixed_timesteps: tuple[int, ...],
  ws: float,
  subsample_steps: int,
) -> dict[str, Any]:
  """Compute task/SMP reward components once per env step and cache them."""
  cache_key = "_task_smp_cache"
  fixed_timesteps = tuple(fixed_timesteps)
  step = int(env.common_step_counter)
  task_terms_signature = _task_terms_signature(task_terms)

  if hasattr(env, cache_key):
    cache = getattr(env, cache_key)
    if (
      cache["step"] == step
      and cache["task_terms_signature"] == task_terms_signature
      and cache["fixed_timesteps"] == fixed_timesteps
      and cache["ws"] == ws
      and cache["subsample_steps"] == subsample_steps
    ):
      return cache

  task_components: dict[str, torch.Tensor] = {}
  task: torch.Tensor | None = None
  for func, weight, kwargs in task_terms:
    component = weight * func(env, **kwargs)
    if task is None:
      task = torch.zeros_like(component)
    task += component
    name = func.__name__
    if name in task_components:
      task_components[name] = task_components[name] + component
    else:
      task_components[name] = component

  if task is None:
    msg = "task_terms must contain at least one task reward component."
    raise ValueError(msg)

  smp = smp_guidance_reward(
    env,
    fixed_timesteps=fixed_timesteps,
    ws=ws,
    subsample_steps=subsample_steps,
  )
  smp_raw_err = getattr(env, "_smp_raw_err", torch.zeros_like(task))

  cache = {
    "step": step,
    "task_terms_signature": task_terms_signature,
    "fixed_timesteps": fixed_timesteps,
    "ws": ws,
    "subsample_steps": subsample_steps,
    "task": task,
    "smp": smp,
    "product": task * smp,
    "sum": task + smp,
    "task_components": task_components,
    "smp_raw_err": smp_raw_err,
  }
  setattr(env, cache_key, cache)
  return cache


def _combine_cached_reward(
  cache: dict[str, Any],
  combine_mode: str,
  task_scale: float = 1.0,
  smp_scale: float = 1.0,
) -> torch.Tensor:
  if combine_mode == "product":
    return cache["product"]
  if combine_mode == "sum":
    return task_scale * cache["task"] + smp_scale * cache["smp"]
  msg = f"Unsupported combine_mode '{combine_mode}'. Expected 'product' or 'sum'."
  raise ValueError(msg)


def combined_reward(
  env: ManagerBasedRlEnv,
  task_terms: tuple[TaskTerm, ...],
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
  combine_mode: str = "product",
  task_scale: float = 1.0,
  smp_scale: float = 1.0,
  subsample_steps: int = 1,
) -> torch.Tensor:
  """Combine task and SMP rewards using either ``product`` or ``sum`` mode."""
  cache = _ensure_task_smp_cache(
    env,
    task_terms=task_terms,
    fixed_timesteps=fixed_timesteps,
    ws=ws,
    subsample_steps=subsample_steps,
  )
  return _combine_cached_reward(
    cache,
    combine_mode,
    task_scale=task_scale,
    smp_scale=smp_scale,
  )


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
  del command_name
  cache = _ensure_task_smp_cache(
    env,
    task_terms=task_terms,
    fixed_timesteps=fixed_timesteps,
    ws=ws,
    subsample_steps=subsample_steps,
  )
  return cache["product"]


def task_reward_metric(
  env: ManagerBasedRlEnv,
  task_terms: tuple[TaskTerm, ...],
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
  subsample_steps: int = 1,
) -> torch.Tensor:
  """Per-step task reward contribution before reward-manager scaling."""
  cache = _ensure_task_smp_cache(
    env,
    task_terms=task_terms,
    fixed_timesteps=fixed_timesteps,
    ws=ws,
    subsample_steps=subsample_steps,
  )
  return cache["task"]


def smp_reward_metric(
  env: ManagerBasedRlEnv,
  task_terms: tuple[TaskTerm, ...],
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
  subsample_steps: int = 1,
) -> torch.Tensor:
  """Per-step SMP guidance reward before reward-manager scaling."""
  cache = _ensure_task_smp_cache(
    env,
    task_terms=task_terms,
    fixed_timesteps=fixed_timesteps,
    ws=ws,
    subsample_steps=subsample_steps,
  )
  return cache["smp"]


def total_reward_metric(
  env: ManagerBasedRlEnv,
  task_terms: tuple[TaskTerm, ...],
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
  combine_mode: str = "product",
  task_scale: float = 1.0,
  smp_scale: float = 1.0,
  subsample_steps: int = 1,
) -> torch.Tensor:
  """Per-step total reward value before reward-manager scaling."""
  cache = _ensure_task_smp_cache(
    env,
    task_terms=task_terms,
    fixed_timesteps=fixed_timesteps,
    ws=ws,
    subsample_steps=subsample_steps,
  )
  return _combine_cached_reward(
    cache,
    combine_mode,
    task_scale=task_scale,
    smp_scale=smp_scale,
  )


def task_term_metric(
  env: ManagerBasedRlEnv,
  term_name: str,
  task_terms: tuple[TaskTerm, ...],
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
  subsample_steps: int = 1,
) -> torch.Tensor:
  """Per-step weighted contribution of a single task reward term."""
  cache = _ensure_task_smp_cache(
    env,
    task_terms=task_terms,
    fixed_timesteps=fixed_timesteps,
    ws=ws,
    subsample_steps=subsample_steps,
  )
  if term_name not in cache["task_components"]:
    available = ", ".join(sorted(cache["task_components"]))
    msg = f"Unknown task reward term '{term_name}'. Available: {available}"
    raise ValueError(msg)
  return cache["task_components"][term_name]


def smp_raw_err_metric(
  env: ManagerBasedRlEnv,
  task_terms: tuple[TaskTerm, ...],
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
  subsample_steps: int = 1,
) -> torch.Tensor:
  """Per-step mean raw SMP denoising error before exponentiation."""
  cache = _ensure_task_smp_cache(
    env,
    task_terms=task_terms,
    fixed_timesteps=fixed_timesteps,
    ws=ws,
    subsample_steps=subsample_steps,
  )
  return cache["smp_raw_err"]
