"""Unit tests for SanityFrameTracker (docs/search_methods.md Step 5's
visual sanity check). Synthetic BEV-shaped arrays, no CARLA needed - same
pattern as tests/test_attacks.py."""
import numpy as np
import pytest

from avredteam_carla.attacks.layout import DEFAULT_LAYOUT
from avredteam_carla.attacks.sanity_frames import SanityFrameTracker, default_worst_moment_proxy


def _bev(tick: int) -> np.ndarray:
    # Distinguishable per tick (not just zeros) so identity checks below
    # can tell frames apart.
    arr = np.zeros((DEFAULT_LAYOUT.num_channels, DEFAULT_LAYOUT.width_px, DEFAULT_LAYOUT.width_px), dtype=np.uint8)
    arr[DEFAULT_LAYOUT.road, 0, 0] = tick % 255
    return arr


def test_worst_moment_proxy_collision_dominates():
    collided = default_worst_moment_proxy(steer=0.0, prev_steer=0.0, brake=0.0, collided_this_tick=True)
    heavy_brake_no_collision = default_worst_moment_proxy(steer=0.0, prev_steer=0.0, brake=1.0, collided_this_tick=False)
    assert collided > heavy_brake_no_collision


def test_worst_moment_proxy_handles_no_prev_steer():
    # First tick: prev_steer is None, must not raise.
    score = default_worst_moment_proxy(steer=0.5, prev_steer=None, brake=0.0, collided_this_tick=False)
    assert score == 0.0


def test_start_frame_is_always_tick_zero():
    tracker = SanityFrameTracker()
    for t in range(10):
        tracker.observe(t, _bev(t), _bev(t), steer=0.0, brake=0.0, collided_this_tick=False)
    assert tracker._start[0] == 0


def test_worst_frame_tracks_max_proxy_score():
    tracker = SanityFrameTracker()
    for t in range(10):
        # A collision only at tick 5 - the proxy should pick it out as worst.
        tracker.observe(t, _bev(t), _bev(t), steer=0.0, brake=0.0, collided_this_tick=(t == 5))
    assert tracker._worst[0] == 5


def test_midpoint_doubling_converges_within_factor_of_two(tmp_path):
    tracker = SanityFrameTracker()
    n_ticks = 100
    for t in range(n_ticks):
        tracker.observe(t, _bev(t), _bev(t), steer=0.0, brake=0.0, collided_this_tick=False)
    midpoint_tick = tracker._midpoint[0]
    # Doubling (1, 2, 4, ..., 64) means the last candidate before n_ticks-1
    # is somewhere in [n_ticks/4, n_ticks/2] roughly - loose bound, just
    # confirming it's not stuck at tick 0 or all the way at the end.
    assert n_ticks // 8 <= midpoint_tick < n_ticks


def test_finalize_writes_three_frame_pairs(tmp_path):
    tracker = SanityFrameTracker()
    for t in range(20):
        tracker.observe(t, _bev(t), _bev(t), steer=float(t) * 0.05, brake=0.0, collided_this_tick=(t == 10))
    written = tracker.finalize(tmp_path)
    labels = {label for label, _ in written}
    assert labels == {"start", "midpoint", "worst"}
    for label in ("start", "midpoint", "worst"):
        files = list((tmp_path / label).glob("*.jpg"))
        assert len(files) == 2  # clean + attacked


def test_finalize_handles_very_short_episode(tmp_path):
    """A 1-tick episode: start and worst are both captured (same tick),
    midpoint's first doubling check (tick >= 1) also fires at tick 0? No -
    tick 0 doesn't reach next_check=1, so midpoint stays uncaptured. Must
    not crash either way."""
    tracker = SanityFrameTracker()
    tracker.observe(0, _bev(0), _bev(0), steer=0.0, brake=0.0, collided_this_tick=False)
    written = tracker.finalize(tmp_path)
    labels = {label for label, _ in written}
    assert "start" in labels
    assert "worst" in labels
