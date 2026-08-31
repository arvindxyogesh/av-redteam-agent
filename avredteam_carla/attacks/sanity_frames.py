"""Phase 4 Step 5's visual sanity check: N fixed representative frames per
trial (start, midpoint, worst-moment-by-severity-contribution) rather than
a full-episode frame dump - dumping every tick's BEV pair across hundreds
of Phase 4 trials would be both slow and unnecessary; the point is a human
being able to eyeball a handful of frames per trial, not archive the whole
episode. Distinct from Phase 2's --bev-frames-every (still available,
still periodic, still PNG) - this is Phase 4's own always-on-for-attacked-
trials mechanism.

Only one candidate frame per named slot is buffered in memory at a time
(~1.1MB for a clean+attacked BEV raster pair at the current 15x192x192
layout - see attacks/layout.py), never the whole episode.

Two things aren't knowable while the episode is still running: the final
tick count (needed for a true midpoint) and which tick will turn out to
have contributed most to severity_score (a whole-episode aggregate, not a
per-tick quantity). Both are approximated online rather than requiring a
second pass:
  - midpoint: a standard streaming trick - keep replacing the "midpoint"
    candidate with the current tick's frame every time the current tick
    index reaches double the previously stored candidate's index (1, 2, 4,
    8, ...). This converges to a tick within a factor of ~2 of the true
    final midpoint without ever needing to know the final episode length
    in advance, and never holds more than one candidate pair at a time.
  - worst-moment: tracked via a per-tick proxy score (see
    default_worst_moment_proxy()) computed only from data already
    available at that tick. This is an approximation of "how much did
    this tick contribute to severity_score," not the real thing -
    documented as such, not claimed to be exact.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from avredteam_carla.attacks.layout import BirdviewLayout, DEFAULT_LAYOUT
from avredteam_carla.attacks.visualize import masks_to_rgb, save_png


def default_worst_moment_proxy(
    steer: float, prev_steer: Optional[float], brake: float, collided_this_tick: bool, dt: float = 0.1
) -> float:
    """Weighted to roughly track severity_score's own emphasis (docs/
    evaluator.md #7): a collision this tick dominates, matching
    severity_score's own collided*40 dominant term; abrupt steering and
    heavy braking contribute less, matching the smaller per-term weights
    there. Not a claim that this equals any real per-tick decomposition of
    severity_score - severity_score's own terms (chattering_rate,
    off_lane_frac, ...) are whole-episode aggregates that don't decompose
    into a single tick's contribution at all.
    """
    steering_rate = 0.0 if prev_steer is None else abs(steer - prev_steer) / dt
    return (50.0 if collided_this_tick else 0.0) + steering_rate + brake


class SanityFrameTracker:
    """Fed one observe() call per tick from run_episode()'s tick loop
    (only while an attack + sanity_frames_dir are active - see
    run_clean_episode.py); finalize() writes whatever was captured."""

    def __init__(self, worst_moment_proxy: Callable = default_worst_moment_proxy):
        self._worst_moment_proxy = worst_moment_proxy
        self._start = None  # (tick, clean_bev, attacked_bev)
        self._midpoint = None
        self._midpoint_next_check = 1
        self._worst = None
        self._worst_score = float("-inf")
        self._prev_steer: Optional[float] = None

    def observe(
        self,
        tick: int,
        clean_bev: np.ndarray,
        attacked_bev: np.ndarray,
        steer: float,
        brake: float,
        collided_this_tick: bool,
    ) -> None:
        if self._start is None:
            self._start = (tick, clean_bev.copy(), attacked_bev.copy())

        if tick >= self._midpoint_next_check:
            self._midpoint = (tick, clean_bev.copy(), attacked_bev.copy())
            self._midpoint_next_check *= 2

        score = self._worst_moment_proxy(steer, self._prev_steer, brake, collided_this_tick)
        if score > self._worst_score:
            self._worst_score = score
            self._worst = (tick, clean_bev.copy(), attacked_bev.copy())

        self._prev_steer = steer

    def finalize(self, out_dir, layout: BirdviewLayout = DEFAULT_LAYOUT) -> list:
        """Writes whichever of start/midpoint/worst were actually captured
        (a very short episode may never reach the first doubling check) as
        tick_XXXXXX_{clean,attacked}.jpg pairs under
        out_dir/{start,midpoint,worst}/. save_png() is extension-agnostic
        (both its cv2 and Pillow backends infer format from the path
        suffix) despite the name, so a .jpg path here just works. Returns
        the list of (label, tick) pairs actually written, for logging.
        """
        out_dir = Path(out_dir)
        written = []
        for label, candidate in (("start", self._start), ("midpoint", self._midpoint), ("worst", self._worst)):
            if candidate is None:
                continue
            tick, clean_bev, attacked_bev = candidate
            frame_dir = out_dir / label
            frame_dir.mkdir(parents=True, exist_ok=True)
            save_png(masks_to_rgb(clean_bev, layout), frame_dir / f"tick_{tick:06d}_clean.jpg")
            save_png(masks_to_rgb(attacked_bev, layout), frame_dir / f"tick_{tick:06d}_attacked.jpg")
            written.append((label, tick))
        return written
