"""Turns a raw episode log (the JSON format Phase 1/2's episode runner
produces, extended in Phase 3 with ground-truth lateral_offset_m/
lane_half_width_m/nearest_actor_distance_m per tick - see docs/evaluator.md
#3/#6 for why those needed adding) into the formal EpisodeMetrics defined in
docs/evaluator.md.

Pure Python/stdlib - no CARLA needed, so this is testable directly (see
tests/test_evaluator.py) against synthetic logs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

# Fixed simulator control-loop rate (see docs/setup.md - CARLA's
# fixed_delta_seconds=0.1, synchronous mode). Every metric here is computed
# from every tick at this rate - see docs/evaluator.md #0/#1 for why that
# matters for the aliasing argument.
DT_S = 0.1


@dataclass(frozen=True)
class EpisodeMetrics:
    # Required fields (Phase 3 brief, Step 2)
    chattering_rate: float
    max_steering_jerk: float
    mean_abs_steering_rate: float
    max_lateral_offset: Optional[float]
    off_lane_frac: Optional[float]
    min_obstacle_clearance: Optional[float]
    collided: bool
    severity_score: float
    # Supporting fields - needed to justify severity_score and to fill in
    # the acceptance table's "Time-to-collision"/"Max brake" columns (see
    # docs/evaluator.md's closing note).
    time_to_collision_s: Optional[float]
    completed: bool
    max_brake: float
    mean_brake: float
    max_brake_rate: float
    n_ticks: int

    def to_dict(self) -> dict:
        return asdict(self)


def _consecutive_diffs(values: list, dt: float = DT_S) -> list:
    return [(b - a) / dt for a, b in zip(values, values[1:])]


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _chattering_rate(steering_rate: list) -> float:
    """docs/evaluator.md #1: sign flips counted only between consecutive
    *nonzero* rate values, divided by the total number of consecutive pairs
    compared (zero-rate pairs count toward the denominator, never the
    numerator - a flat/unchanging steering signal can't "flip")."""
    if len(steering_rate) < 2:
        return 0.0
    pairs = list(zip(steering_rate[:-1], steering_rate[1:]))
    flips = 0
    for a, b in pairs:
        sa, sb = _sign(a), _sign(b)
        if sa != 0 and sb != 0 and sa != sb:
            flips += 1
    return flips / len(pairs)


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _max_abs(values: list) -> float:
    return max((abs(v) for v in values), default=0.0)


def evaluate(log: dict) -> EpisodeMetrics:
    ticks = log["ticks"]
    n_ticks = len(ticks)

    steer = [t["steer"] for t in ticks]
    brake = [t["brake"] for t in ticks]

    steering_rate = _consecutive_diffs(steer)
    steering_jerk = _consecutive_diffs(steering_rate)
    chattering_rate = _chattering_rate(steering_rate)
    max_steering_jerk = _max_abs(steering_jerk)
    mean_abs_steering_rate = _mean([abs(r) for r in steering_rate])

    max_brake = max(brake, default=0.0)
    mean_brake = _mean(brake)
    brake_rate = _consecutive_diffs(brake)
    max_brake_rate = _max_abs(brake_rate)

    # Ground-truth fields added in Phase 3 (docs/evaluator.md #3/#6). None
    # of these existed in Phase 1/2 logs, so handle their absence
    # gracefully (evaluate() should still run against an older log, just
    # with these fields coming back as None) rather than KeyError.
    lateral_offsets = [t.get("lateral_offset_m") for t in ticks]
    lane_half_widths = [t.get("lane_half_width_m") for t in ticks]
    nearest_actor_distances = [t.get("nearest_actor_distance_m") for t in ticks]

    have_lateral = all(v is not None for v in lateral_offsets) and n_ticks > 0
    max_lateral_offset = max(lateral_offsets) if have_lateral else None

    off_lane_frac: Optional[float]
    if have_lateral and all(v is not None for v in lane_half_widths):
        off_lane_ticks = sum(
            1 for lo, lhw in zip(lateral_offsets, lane_half_widths) if lo > lhw
        )
        off_lane_frac = off_lane_ticks / n_ticks
    else:
        off_lane_frac = None

    have_clearance = all(v is not None for v in nearest_actor_distances) and n_ticks > 0
    min_obstacle_clearance = min(nearest_actor_distances) if have_clearance else None

    # Collision: ground truth, from the collision sensor events Phase 1
    # already logs per tick (docs/setup.md/run_clean_episode.py's
    # _extract_events - falsy on ticks with no event).
    collision_tick_idx = next(
        (i for i, t in enumerate(ticks) if t.get("collision_events")), None
    )
    collided = collision_tick_idx is not None
    time_to_collision_s = collision_tick_idx * DT_S if collided else None

    completed = (not collided) and log.get("meta", {}).get("termination_reason") == "route_completed"

    off_lane_term = off_lane_frac if off_lane_frac is not None else 0.0
    clearance_term = min_obstacle_clearance if min_obstacle_clearance is not None else float("inf")

    severity_score = (
        (40.0 if collided else 0.0)
        + min(25.0, chattering_rate * 25.0)
        + min(15.0, off_lane_term * 15.0)
        + min(10.0, max(0.0, 2.0 - clearance_term) * 5.0)
        + min(10.0, mean_abs_steering_rate * 5.0)
    )
    severity_score = min(100.0, severity_score)

    return EpisodeMetrics(
        chattering_rate=chattering_rate,
        max_steering_jerk=max_steering_jerk,
        mean_abs_steering_rate=mean_abs_steering_rate,
        max_lateral_offset=max_lateral_offset,
        off_lane_frac=off_lane_frac,
        min_obstacle_clearance=min_obstacle_clearance,
        collided=collided,
        severity_score=severity_score,
        time_to_collision_s=time_to_collision_s,
        completed=completed,
        max_brake=max_brake,
        mean_brake=mean_brake,
        max_brake_rate=max_brake_rate,
        n_ticks=n_ticks,
    )
