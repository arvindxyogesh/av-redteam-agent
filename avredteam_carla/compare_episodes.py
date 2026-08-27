"""Compares a clean and an attacked episode log (both produced by
run_clean_episode.py) and reports the deviation metrics the Phase 2
acceptance table asks for: steering-rate sign flips (channel-noise attack),
mean/min ground-truth speed and brake stats (phantom-actor attack), and a
tick-paired control-output divergence useful for any attack type.

Works entirely from the JSON files the runner already writes - no CARLA
needed, so this is runnable anywhere the two log files exist (including
this dev sandbox, unlike the runner itself).

Usage:
    python -m avredteam_carla.compare_episodes \\
        --clean logs/clean_episode.json --attacked logs/attacked_episode.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def sign_flip_count(values: list) -> int:
    """Counts direction reversals in a series (e.g. steer over time) -
    zero-valued differences (no change tick-to-tick) don't break a run of
    the same sign, matching how a human eyeballing "is it oscillating?"
    would read a trace with flat spots in it."""
    diffs = [b - a for a, b in zip(values, values[1:])]
    signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in diffs]
    flips = 0
    prev_sign = 0
    for s in signs:
        if s == 0:
            continue
        if prev_sign != 0 and s != prev_sign:
            flips += 1
        prev_sign = s
    return flips


def episode_metrics(log: dict) -> dict:
    ticks = log["ticks"]
    steer = [t["steer"] for t in ticks]
    brake = [t["brake"] for t in ticks]
    speed = [t["ground_truth_speed"] for t in ticks if t.get("ground_truth_speed") is not None]
    return {
        "n_ticks": len(ticks),
        "termination_reason": log.get("meta", {}).get("termination_reason"),
        "steer_sign_flips": sign_flip_count(steer),
        "mean_brake": (sum(brake) / len(brake)) if brake else None,
        "max_brake": max(brake) if brake else None,
        "mean_speed": (sum(speed) / len(speed)) if speed else None,
        "min_speed": min(speed) if speed else None,
    }


def paired_control_deviation(clean_ticks: list, attacked_ticks: list) -> dict:
    """Tick-by-tick |clean - attacked| control divergence, over the ticks
    both episodes have in common. Two episodes can legitimately have
    different lengths (an attack changing when/whether the route
    completes is itself a valid outcome), so this only compares the
    overlapping prefix and reports how many ticks that was."""
    n = min(len(clean_ticks), len(attacked_ticks))
    if n == 0:
        return {"n_compared": 0}
    abs_steer_diff = [abs(clean_ticks[i]["steer"] - attacked_ticks[i]["steer"]) for i in range(n)]
    abs_brake_diff = [abs(clean_ticks[i]["brake"] - attacked_ticks[i]["brake"]) for i in range(n)]
    return {
        "n_compared": n,
        "mean_abs_steer_diff": sum(abs_steer_diff) / n,
        "max_abs_steer_diff": max(abs_steer_diff),
        "mean_abs_brake_diff": sum(abs_brake_diff) / n,
        "max_abs_brake_diff": max(abs_brake_diff),
    }


def compare(clean_log: dict, attacked_log: dict) -> dict:
    return {
        "clean": episode_metrics(clean_log),
        "attacked": episode_metrics(attacked_log),
        "paired_control_deviation": paired_control_deviation(clean_log["ticks"], attacked_log["ticks"]),
        "attack_meta": attacked_log.get("meta", {}).get("attack"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean", required=True, help="Path to a clean-episode JSON log")
    p.add_argument("--attacked", required=True, help="Path to an attacked-episode JSON log")
    p.add_argument("--out", default=None, help="Optional path to also write the comparison as JSON")
    args = p.parse_args()

    clean_log = json.loads(Path(args.clean).read_text())
    attacked_log = json.loads(Path(args.attacked).read_text())

    result = compare(clean_log, attacked_log)
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
