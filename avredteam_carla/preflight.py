"""Pre-flight check run once before a Phase 4 campaign starts - not before
every trial, that would be wasteful (docs/search_methods.md Step 5).
Snapshots disk headroom, per-GPU utilization, and load average, and picks
an idle/low-utilization GPU automatically rather than hardcoding one -
extending TASK.md's original Phase 1 decision ("GPU choice is
configurable... never hardcoded; prefer an idle GPU") from "a human checks
nvidia-smi once" to "the campaign runner checks it once per campaign."

Motivated directly by docs/evaluator.md's Phase 3 findings: repeated
run_trial() calls crashed in a pattern that tracked real-time CPU load and
disk pressure on this shared node, not a fixed call count or a specific
GPU. Snapshotting conditions once up front (and logging them into
CampaignResult.preflight) lets later analysis check whether a campaign's
anomalies correlated with what the node looked like when it started.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable, List, Optional


def _run_nvidia_smi(cmd: list, timeout: float = 10.0) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
    return result.stdout


def parse_nvidia_smi_csv(output: str) -> List[dict]:
    """Parses the output of:
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \\
            --format=csv,noheader,nounits
    into a list of {"index": int, "utilization_pct": float,
    "memory_used_mb": float, "memory_total_mb": float}, one entry per GPU.
    """
    gpus = []
    for line in output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        index, util, mem_used, mem_total = (p.strip() for p in line.split(","))
        gpus.append({
            "index": int(index),
            "utilization_pct": float(util),
            "memory_used_mb": float(mem_used),
            "memory_total_mb": float(mem_total),
        })
    return gpus


def gpu_snapshot(runner: Callable[[list], str] = _run_nvidia_smi) -> Optional[List[dict]]:
    """None if nvidia-smi isn't available (e.g. this dev sandbox has no
    GPU at all) - a real Maui run always has it; returning None rather
    than raising keeps this module importable and testable everywhere.
    `runner` is injectable so tests never actually spawn nvidia-smi.
    """
    try:
        output = runner([
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return parse_nvidia_smi_csv(output)


def disk_snapshot(path: str = "/") -> dict:
    total, used, free = shutil.disk_usage(path)
    return {
        "path": path,
        "total_gb": total / 1e9,
        "used_gb": used / 1e9,
        "free_gb": free / 1e9,
        "used_pct": 100.0 * used / total if total else 0.0,
    }


def load_average_snapshot() -> Optional[dict]:
    try:
        one, five, fifteen = os.getloadavg()
    except (OSError, AttributeError):
        # AttributeError: getloadavg() doesn't exist on this platform (e.g.
        # Windows) - not expected on Maui, but shouldn't crash elsewhere.
        return None
    return {"load_1m": one, "load_5m": five, "load_15m": fifteen}


def pick_idle_gpu(gpus: List[dict], exclude: Optional[List[int]] = None) -> Optional[int]:
    """Picks the GPU with the lowest utilization_pct, ties broken by
    lowest memory_used_mb - "prefer an idle GPU" (TASK.md's original
    Phase 1 instruction), automated here for Phase 4's campaign runner
    rather than a human eyeballing nvidia-smi each time. None if `gpus` is
    empty (e.g. gpu_snapshot() returned None/[]) or every GPU is excluded.
    """
    candidates = [g for g in gpus if exclude is None or g["index"] not in exclude]
    if not candidates:
        return None
    return min(candidates, key=lambda g: (g["utilization_pct"], g["memory_used_mb"]))["index"]


def preflight_snapshot(disk_path: str = "/", gpu_runner: Callable[[list], str] = _run_nvidia_smi) -> dict:
    """The full snapshot Step 5 asks be logged into CampaignResult.preflight
    (or a sidecar file) before a campaign's first trial runs."""
    gpus = gpu_snapshot(gpu_runner)
    return {
        "disk": disk_snapshot(disk_path),
        "gpus": gpus,
        "load_average": load_average_snapshot(),
        "picked_gpu": pick_idle_gpu(gpus) if gpus else None,
    }
