"""Generate a motion window with a trained SMP diffusion model and visualize
the predicted trajectory in a viser viewer.

Features carry ``root_pos`` (xy heading-inv + world z) and ``root_rot``
(6D tan-norm, heading-inv relative to the last-frame root), so the
world-frame pelvis trajectory is reconstructed directly from those two —
no velocity integration needed.  The last window frame is placed at a
chosen anchor pose (default: the robot's default standing state) and the
rest of the window is reconstructed relative to it.  EE positions come
from the sampled ``ee_pos`` feature lifted into world via the per-frame
pelvis pose.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import viser
from mjlab.entity import Entity
from mjlab.viewer.viser.scene import MjlabViserScene

from smp.pretrain.inference import (
  build_model_and_scheduler,
  resolve_style_label,
  sample_window_composed,
  sample_window,
)
from smp.sampling.feature_to_state import (
  NUM_EE,
  window_to_ee_trajectories,
  window_to_pelvis_trajectory,
)
from smp.utils import detect_device


@dataclass
class Cfg:
  ckpt_path: str = ""
  """Path to a local SMP diffusion checkpoint .pt file. Mutually exclusive with --wandb-run."""
  wandb_run: str = ""
  """W&B run path '<entity>/<project>/<run_id>'. Downloads the latest .pt from the run."""
  device: str = ""
  """Compute device. Empty = auto."""
  fps: float = 50.0
  """Playback frame rate."""
  style: str = ""
  """Optional conditioned style name. Empty keeps unconditional sampling."""
  style_upper: str = ""
  """Upper-body style label for composed sampling."""
  style_lower: str = ""
  """Lower-body style label for composed sampling."""
  cfg_scale: float = 1.0
  """Classifier-free guidance scale for conditioned priors."""
  sampler: str = "ddpm"
  """Sampling algorithm: ``ddpm`` or ``ddim``."""
  num_steps: int = 20
  """Number of DDIM steps. Ignored when sampler=ddpm."""


def _resolve_ckpt_path(cfg: Cfg) -> str:
  """Return a local ckpt path, downloading from wandb if --wandb-run is set."""
  if bool(cfg.ckpt_path) == bool(cfg.wandb_run):
    msg = "Specify exactly one of --ckpt-path or --wandb-run"
    raise ValueError(msg)
  if cfg.ckpt_path:
    return cfg.ckpt_path

  import wandb

  api = wandb.Api()
  run = api.run(cfg.wandb_run)
  pt_files = [f for f in run.files() if f.name.endswith(".pt")]
  if not pt_files:
    msg = f"No .pt files in wandb run {cfg.wandb_run}"
    raise FileNotFoundError(msg)
  target = next(
    (f for f in pt_files if Path(f.name).name == "pretrained.pt"),
    sorted(pt_files, key=lambda f: f.name)[-1],
  )
  download_dir = Path("logs") / "wandb_ckpt_cache" / cfg.wandb_run.replace("/", "_")
  download_dir.mkdir(parents=True, exist_ok=True)
  target.download(root=str(download_dir), replace=True)
  local = download_dir / target.name
  print(f"Downloaded {target.name} from {cfg.wandb_run} -> {local}")
  return str(local)


def _setup_g1_sim(device: str):
  """Build a single G1 sim. Mirrors scripts/csv_to_npz.py:_setup_sim."""
  from mjlab.scene import Scene
  from mjlab.sim.sim import Simulation, SimulationCfg
  from mjlab.tasks.tracking.config.g1.env_cfgs import (
    unitree_g1_flat_tracking_env_cfg,
  )

  sim_cfg = SimulationCfg()
  env_cfg = unitree_g1_flat_tracking_env_cfg()
  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return sim, scene


def _write_pose_to_robot(
  robot: Entity,
  pelvis_pos: np.ndarray,
  pelvis_quat_wxyz: np.ndarray,
  joint_pos: np.ndarray,
  device: str,
) -> None:
  """Mirror scripts/csv_to_npz.py:_fk_motion's per-frame state write."""
  root = robot.data.default_root_state.clone()
  root[:, 0:3] = torch.as_tensor(pelvis_pos, device=device, dtype=root.dtype)
  root[:, 3:7] = torch.as_tensor(pelvis_quat_wxyz, device=device, dtype=root.dtype)
  robot.write_root_state_to_sim(root)
  jp = robot.data.default_joint_pos.clone()
  jp[:] = torch.as_tensor(joint_pos, device=device, dtype=jp.dtype)
  jv = robot.data.default_joint_vel.clone()
  robot.write_joint_state_to_sim(jp, jv)


def main(cfg: Cfg) -> None:
  device_str = cfg.device or detect_device()
  device = torch.device(device_str)
  print(f"Device: {device_str}")

  ckpt_path = _resolve_ckpt_path(cfg)
  ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
  model, scheduler, q_low, q_high = build_model_and_scheduler(ckpt, device)
  print(f"Loaded checkpoint epoch={ckpt.get('epoch')} from {ckpt_path}")

  feature_dim = int(ckpt["cfg"]["feature_dim"])
  window_size = int(ckpt["cfg"]["window_size"])
  style_names = tuple(ckpt["cfg"].get("style_names", ()))
  style_id = None
  style_upper_id = None
  style_lower_id = None
  has_composed_style = bool(cfg.style_upper) or bool(cfg.style_lower)
  if cfg.style and has_composed_style:
    msg = "Use either --style or both --style-upper/--style-lower, not both."
    raise ValueError(msg)
  if cfg.style:
    if not style_names:
      msg = "Checkpoint is unconditional; --style is not supported."
      raise ValueError(msg)
    style_id = resolve_style_label(style_names, cfg.style)
    print(f"Sampling conditioned style '{cfg.style}' (id={style_id}), cfg_scale={cfg.cfg_scale}")
  elif has_composed_style:
    if not style_names:
      msg = "Checkpoint is unconditional; composed style sampling is not supported."
      raise ValueError(msg)
    if not cfg.style_upper or not cfg.style_lower:
      msg = "Composed sampling requires both --style-upper and --style-lower."
      raise ValueError(msg)
    style_upper_id = resolve_style_label(style_names, cfg.style_upper)
    style_lower_id = resolve_style_label(style_names, cfg.style_lower)
    print(
      "Sampling composed style "
      f"upper='{cfg.style_upper}' (id={style_upper_id}), "
      f"lower='{cfg.style_lower}' (id={style_lower_id}), "
      f"cfg_scale={cfg.cfg_scale}, sampler={cfg.sampler}"
    )

  sim_device = device_str
  sim, scene = _setup_g1_sim(sim_device)
  robot: Entity = scene["robot"]
  mj_model = sim.mj_model

  # Place the last window frame at the robot's default standing pose.
  anchor_pelvis_pos = robot.data.default_root_state[0, 0:3].detach().cpu()
  anchor_pelvis_quat = robot.data.default_root_state[0, 3:7].detach().cpu()

  def run() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if style_upper_id is not None and style_lower_id is not None:
      pred_denorm = sample_window_composed(
        model,
        scheduler,
        q_low,
        q_high,
        window_size,
        feature_dim,
        device,
        style_upper_id=style_upper_id,
        style_lower_id=style_lower_id,
        cfg_scale=cfg.cfg_scale,
        sampler=cfg.sampler,
        num_steps=cfg.num_steps if cfg.sampler.lower() == "ddim" else None,
      )
    else:
      pred_denorm = sample_window(
        model,
        scheduler,
        q_low,
        q_high,
        window_size,
        feature_dim,
        device,
        style_id=style_id,
        cfg_scale=cfg.cfg_scale,
        sampler=cfg.sampler,
        num_steps=cfg.num_steps if cfg.sampler.lower() == "ddim" else None,
      )
    p_pos, p_quat, p_joint = window_to_pelvis_trajectory(
      pred_denorm,
      anchor_pelvis_pos,
      anchor_pelvis_quat,
    )
    ee_pos = window_to_ee_trajectories(pred_denorm, p_pos, p_quat)
    return (
      p_pos.cpu().numpy(),
      p_quat.cpu().numpy(),
      p_joint.cpu().numpy(),
      ee_pos.cpu().numpy(),
    )

  state: dict = {"pred": run()}

  server = viser.ViserServer()
  viser_scene = MjlabViserScene(server, mj_model, num_envs=1)
  viser_scene.debug_visualization_enabled = True

  # /fixed_bodies parents under mjviser's camera-tracking scene offset, so
  # the points stay aligned with the re-centered robot.
  ee_points = server.scene.add_point_cloud(
    name="/fixed_bodies/predicted_ee_positions",
    points=np.zeros((NUM_EE, 3), dtype=np.float32),
    colors=np.tile(np.array([255, 80, 0], dtype=np.uint8), (NUM_EE, 1)),
    point_size=0.03,
  )

  with server.gui.add_folder("Generate"):
    frame_slider = server.gui.add_slider(
      "Frame", min=0, max=window_size - 1, step=1, initial_value=0
    )
    play_btn = server.gui.add_button("Play / Pause")
    resample_btn = server.gui.add_button("Resample")

  playing = {"v": True}

  @play_btn.on_click
  def _(_evt) -> None:
    playing["v"] = not playing["v"]

  @resample_btn.on_click
  def _(_evt) -> None:
    state["pred"] = run()

  def render(frame: int) -> None:
    p_pos, p_quat, p_joint, ee_pos = state["pred"]
    _write_pose_to_robot(robot, p_pos[frame], p_quat[frame], p_joint[frame], sim_device)
    sim.forward()
    wd = sim.wp_data
    viser_scene.update_from_arrays(
      body_xpos=np.asarray(wd.xpos.numpy()),
      body_xmat=np.asarray(wd.xmat.numpy()),
      qpos=np.asarray(wd.qpos.numpy()),
      env_idx=0,
    )
    ee_points.points = ee_pos[frame]
    viser_scene.refresh_visualization()

  print("Viser server running. Open the printed URL.")
  dt_play = 1.0 / cfg.fps
  try:
    while True:
      render(int(frame_slider.value))
      if playing["v"]:
        nxt = (int(frame_slider.value) + 1) % window_size
        frame_slider.value = nxt
      time.sleep(dt_play)
  except KeyboardInterrupt:
    print("Shutting down.")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
