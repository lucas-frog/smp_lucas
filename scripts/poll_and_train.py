#!/usr/bin/env python3
"""Poll local GPUs and launch a training command when one is free.

Runs directly on the GPU server.  Usage::

  tmux new -s poll
  uv run scripts/poll_and_train.py --name velocity -- \
    uv run scripts/train.py Smp-Velocity-G1 --env.scene.num-envs=4096
  # Ctrl+B D to detach

A GPU is free when compute util ≤ *max-util* AND memory used ≤ *max-memory-mb*.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _query_gpus() -> list[tuple[int, float, int, int]]:
    """Return [(index, util%, mem_used_mb, mem_total_mb), ...]."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True, check=True,
    ).stdout

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpus.append((int(parts[0]), float(parts[1]), int(parts[2]), int(parts[3])))
    return gpus


def find_free_gpu(
    max_util: float, max_mem_mb: int, allowed: set[int] | None
) -> int | None:
    """Return the index of the first free GPU, or None."""
    for idx, util, mem_used, _ in _query_gpus():
        if allowed is not None and idx not in allowed:
            continue
        if util <= max_util and mem_used <= max_mem_mb:
            return idx
    return None


def gpu_summary(allowed: set[int] | None) -> str:
    parts = []
    for idx, util, mem_used, mem_total in _query_gpus():
        if allowed is not None and idx not in allowed:
            continue
        free = mem_total - mem_used
        parts.append(
            f"  GPU {idx}: {util:5.0f}% util, {mem_used:5d}/{mem_total} MB "
            f"({free} MB free)"
        )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poll GPUs and launch training when free",
        usage=(
            "poll_and_train.py [--interval N] [--max-util N] [--gpus 0,1] "
            "[--name NAME] -- CMD [ARGS...]"
        ),
    )
    parser.add_argument("--interval", type=int, default=30,
                        help="Seconds between polls (default: 30)")
    parser.add_argument("--max-util", type=float, default=5.0,
                        help="Max GPU utilisation %% to consider idle (default: 5)")
    parser.add_argument("--max-memory-mb", type=int, default=2000,
                        help="Max GPU memory (MB) to consider idle (default: 2000)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU indices to consider, e.g. '0,1,2'")
    parser.add_argument("--name", default="train",
                        help="Run name for the log file (default: train)")
    parser.add_argument("--once", action="store_true",
                        help="Check once and exit without polling")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the command but don't launch it")
    parser.add_argument("cmd", nargs=argparse.REMAINDER,
                        help="Training command (place after --)")
    args = parser.parse_args()

    cmd_parts = args.cmd
    if not cmd_parts or cmd_parts[0] == "--":
        # argparse.REMAINDER includes the leading "--" if it's there; strip it
        cmd_parts = cmd_parts[1:]
    if not cmd_parts:
        print("Error: no training command given. Place it after --, e.g.:")
        print("  poll_and_train.py --interval 30 -- uv run ...")
        return

    allowed = {int(x.strip()) for x in args.gpus.split(",")} if args.gpus else None

    print(f"[{_now()}] Polling every {args.interval}s | "
          f"util≤{args.max_util}%, mem≤{args.max_memory_mb}MB"
          + (f" | GPUs: {sorted(allowed)}" if allowed else ""))
    print(f"[{_now()}] Command: {' '.join(cmd_parts)}")
    print(f"[{_now()}] Current GPU state:")
    print(gpu_summary(allowed))
    print()

    while True:
        gpu = find_free_gpu(args.max_util, args.max_memory_mb, allowed)

        if gpu is not None:
            free_mem = [
                f"{mem_total - mem_used} MB"
                for idx, _, mem_used, mem_total in _query_gpus()
                if idx == gpu
            ][0]
            print(f"[{_now()}] GPU {gpu} is free ({free_mem} free)")

            log_file = f"~/log_{args.name}.log"
            full_cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} "
                f"nohup {' '.join(cmd_parts)} "
                f"> {log_file} 2>&1 &"
            )

            if args.dry_run:
                print(f"  [DRY RUN] Would execute:\n  {full_cmd}")
            else:
                subprocess.Popen(full_cmd, shell=True, start_new_session=True)
                print(f"  Launched! Monitor: tail -f {log_file}")
            return

        print(f"[{_now()}] No free GPU, retrying in {args.interval}s...")
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
