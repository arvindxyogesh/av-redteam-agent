"""Unit tests for avredteam_carla.analyze_episode_length_bias
(docs/search_methods.md Step 0). Locks in the module docstring's central
claim on synthetic data: min_obstacle_clearance/max_lateral_offset are
structurally length-biased (running extrema over the whole episode) in a
way off_lane_frac/chattering_rate are not (already rate-normalized) - a
property of avredteam_carla.evaluator.evaluate() itself, verifiable
without any real hardware or CARLA log."""
import random

import pytest

from avredteam_carla.analyze_episode_length_bias import (
    CANDIDATE_METRICS,
    analyze_prefix_length_sensitivity,
    pearson_r,
    truncate_log,
)
from tests.test_evaluator import make_log


def _random_walk_log(n=400, seed=0):
    """A log whose ground-truth fields wander like real driving data would
    (not monotonic by construction) - so any monotonicity found in the
    analysis below comes from evaluate()'s min/max, not from a rigged
    input series."""
    rng = random.Random(seed)
    steer = [0.0]
    lateral_offset_m = [0.5]
    nearest_actor_distance_m = [15.0]
    for _ in range(n - 1):
        steer.append(max(-1.0, min(1.0, steer[-1] + rng.uniform(-0.1, 0.1))))
        lateral_offset_m.append(max(0.0, lateral_offset_m[-1] + rng.uniform(-0.3, 0.3)))
        nearest_actor_distance_m.append(max(0.1, nearest_actor_distance_m[-1] + rng.uniform(-1.0, 1.0)))
    return make_log(
        steer=steer,
        lateral_offset_m=lateral_offset_m,
        lane_half_width_m=[1.75] * n,
        nearest_actor_distance_m=nearest_actor_distance_m,
    )


# --- pearson_r -------------------------------------------------------


def test_pearson_r_perfect_positive_correlation():
    assert pearson_r([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_pearson_r_perfect_negative_correlation():
    assert pearson_r([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_pearson_r_none_for_zero_variance():
    assert pearson_r([1, 2, 3], [5, 5, 5]) is None


def test_pearson_r_none_for_too_few_points():
    assert pearson_r([1], [1]) is None


# --- truncate_log ----------------------------------------------------


def test_truncate_log_keeps_only_first_n_ticks():
    log = _random_walk_log(n=100)
    truncated = truncate_log(log, 30)
    assert len(truncated["ticks"]) == 30
    assert truncated["ticks"] == log["ticks"][:30]
    assert len(log["ticks"]) == 100  # original untouched


# --- the central structural-bias finding ----------------------------


def test_min_obstacle_clearance_is_non_increasing_as_prefix_grows():
    """A min over a longer prefix of the same series can only stay equal
    or decrease - the mathematical certainty the module docstring makes,
    checked directly rather than asserted."""
    log = _random_walk_log(n=300, seed=1)
    result = analyze_prefix_length_sensitivity(log, fractions=(0.1, 0.2, 0.3, 0.5, 0.7, 1.0))
    values = [row["min_obstacle_clearance"] for row in result["rows"]]
    assert values == sorted(values, reverse=True)  # non-increasing


def test_max_lateral_offset_is_non_decreasing_as_prefix_grows():
    log = _random_walk_log(n=300, seed=2)
    result = analyze_prefix_length_sensitivity(log, fractions=(0.1, 0.2, 0.3, 0.5, 0.7, 1.0))
    values = [row["max_lateral_offset"] for row in result["rows"]]
    assert values == sorted(values)  # non-decreasing


def test_min_obstacle_clearance_correlates_negatively_with_length_on_average():
    """Not just one lucky seed - the monotonicity property holds across
    several independent random-walk logs, so the resulting correlation
    with n_ticks is reliably negative (not just occasionally)."""
    correlations = []
    for seed in range(8):
        log = _random_walk_log(n=250, seed=seed)
        result = analyze_prefix_length_sensitivity(log, fractions=(0.2, 0.4, 0.6, 0.8, 1.0))
        r = result["correlation_with_n_ticks"]["min_obstacle_clearance"]
        if r is not None:
            correlations.append(r)
    assert len(correlations) >= 6
    assert all(r <= 0 for r in correlations)


def test_max_lateral_offset_correlates_positively_with_length_on_average():
    correlations = []
    for seed in range(8):
        log = _random_walk_log(n=250, seed=seed)
        result = analyze_prefix_length_sensitivity(log, fractions=(0.2, 0.4, 0.6, 0.8, 1.0))
        r = result["correlation_with_n_ticks"]["max_lateral_offset"]
        if r is not None:
            correlations.append(r)
    assert len(correlations) >= 6
    assert all(r >= 0 for r in correlations)


def test_off_lane_frac_is_not_monotonic_in_prefix_length():
    """Contrast case: off_lane_frac is a fraction (already normalized by
    n_ticks), so - unlike the two extrema above - it must NOT be forced
    monotonic by construction. At least one seed among several should show
    a non-monotonic trace, confirming this isn't the same structural bias."""
    saw_non_monotonic = False
    for seed in range(15):
        log = _random_walk_log(n=200, seed=seed)
        result = analyze_prefix_length_sensitivity(log, fractions=(0.2, 0.4, 0.6, 0.8, 1.0))
        values = [row["off_lane_frac"] for row in result["rows"]]
        if values != sorted(values) and values != sorted(values, reverse=True):
            saw_non_monotonic = True
            break
    assert saw_non_monotonic


def test_analyze_prefix_length_sensitivity_report_shape():
    log = _random_walk_log(n=100, seed=0)
    result = analyze_prefix_length_sensitivity(log, fractions=(0.5, 1.0))
    assert result["n_ticks_total"] == 100
    assert len(result["rows"]) == 2
    assert set(CANDIDATE_METRICS) <= set(result["correlation_with_n_ticks"])


def test_analyze_handles_log_without_ground_truth_fields():
    """A Phase 1/2-vintage log (no lateral_offset_m/nearest_actor_distance_m)
    must not crash - those fields just come back as None, per
    evaluate()'s own documented graceful-absence handling."""
    rng = random.Random(0)
    steer = [0.0]
    for _ in range(49):
        steer.append(max(-1.0, min(1.0, steer[-1] + rng.uniform(-0.2, 0.2))))
    log = make_log(steer=steer)  # no ground-truth kwargs
    result = analyze_prefix_length_sensitivity(log, fractions=(0.5, 1.0))
    assert result["correlation_with_n_ticks"]["min_obstacle_clearance"] is None
    assert result["correlation_with_n_ticks"]["chattering_rate"] is not None
