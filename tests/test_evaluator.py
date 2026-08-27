"""Unit tests for the Phase 3 evaluator - synthetic logs matching the real
schema (Phase 1/2's runner + Phase 3's two new ground-truth fields), no
CARLA needed. See docs/evaluator.md for the formulas being verified here."""
import math

from avredteam_carla.evaluator import evaluate, EpisodeMetrics, DT_S


def make_log(
    steer,
    brake=None,
    lateral_offset_m=None,
    lane_half_width_m=None,
    nearest_actor_distance_m=None,
    collision_at=None,
    termination_reason="route_completed",
):
    n = len(steer)
    brake = brake or [0.0] * n
    ticks = []
    for i in range(n):
        tick = {
            "tick": i,
            "steer": steer[i],
            "throttle": 0.5,
            "brake": brake[i],
            "collision_events": [{"collision_type": 1}] if collision_at == i else [],
        }
        if lateral_offset_m is not None:
            tick["lateral_offset_m"] = lateral_offset_m[i]
        if lane_half_width_m is not None:
            tick["lane_half_width_m"] = lane_half_width_m[i]
        if nearest_actor_distance_m is not None:
            tick["nearest_actor_distance_m"] = nearest_actor_distance_m[i]
        ticks.append(tick)
    return {"meta": {"termination_reason": termination_reason}, "ticks": ticks}


# ---------------------------------------------------------------------------
# chattering_rate
# ---------------------------------------------------------------------------

def test_chattering_rate_zero_for_monotonic_steer():
    log = make_log(steer=[0.0, 0.1, 0.2, 0.3, 0.4])
    m = evaluate(log)
    assert m.chattering_rate == 0.0


def test_chattering_rate_counts_only_nonzero_pair_flips():
    # rates: +0.5, -0.5, +0.5, -0.5 -> 3 flips out of 4 pairs of rates
    # (pairs of *rates*, i.e. len(rates)-1 = 3 comparisons)
    steer = [0.0, 0.5, 0.0, 0.5, 0.0]
    log = make_log(steer=steer)
    m = evaluate(log)
    # rates = [+5, -5, +5, -5] (per-second, dt=0.1) -> 3 consecutive pairs, all flips
    assert m.chattering_rate == 1.0


def test_chattering_rate_flat_run_does_not_count_as_flip_but_fills_denominator():
    # rates: +5, 0, +5 -> pairs: (+5,0) not a flip (zero involved), (0,+5) not a flip
    # denominator = 2 (total pairs), numerator = 0
    steer = [0.0, 0.5, 0.5, 1.0]
    log = make_log(steer=steer)
    m = evaluate(log)
    assert m.chattering_rate == 0.0


def test_chattering_rate_resolves_2hz_oscillation_at_10hz_sampling():
    """Aliasing-safety sanity check (docs/evaluator.md #1): a 2Hz sinusoidal
    steer trace sampled at 10Hz (5 samples/cycle) should register a clearly
    nonzero chattering_rate, not get silently aliased away to ~0.

    Note the expected magnitude here: a *smooth* sinusoid's rate (its
    derivative, also a sinusoid) only flips sign near its own zero-crossings
    - about twice per cycle - not on every sample, so ~5 samples/cycle gives
    ~2/5 = 0.4, not something close to 1.0. A rate that reverses on every
    single tick (see test_chattering_rate_counts_only_nonzero_pair_flips,
    which gives 1.0) is a *choppier* signal than a clean low-frequency
    oscillation, not a subset of it - both are clearly distinguishable from
    a monotonic/flat trace (0.0), which is what the aliasing check actually
    needs to guarantee.
    """
    n = 100
    steer = [0.5 * math.sin(2 * math.pi * 2.0 * i * DT_S) for i in range(n)]
    log = make_log(steer=steer)
    m = evaluate(log)
    assert m.chattering_rate > 0.3  # clearly nonzero, not aliased to ~0


# ---------------------------------------------------------------------------
# jerk / steering rate
# ---------------------------------------------------------------------------

def test_max_steering_jerk_and_mean_abs_rate_on_simple_ramp():
    steer = [0.0, 0.1, 0.2, 0.3]  # constant rate = 1.0/s, zero jerk
    log = make_log(steer=steer)
    m = evaluate(log)
    assert abs(m.mean_abs_steering_rate - 1.0) < 1e-9
    assert abs(m.max_steering_jerk) < 1e-9


def test_max_steering_jerk_detects_a_sudden_rate_change():
    steer = [0.0, 0.1, 0.2, -0.2]  # rate jumps from 1.0/s to -4.0/s
    log = make_log(steer=steer)
    m = evaluate(log)
    assert m.max_steering_jerk > 0


# ---------------------------------------------------------------------------
# route/lane deviation (ground-truth fields)
# ---------------------------------------------------------------------------

def test_off_lane_frac_and_max_lateral_offset():
    steer = [0.0] * 4
    lateral_offset_m = [0.5, 2.0, 0.3, 3.0]
    lane_half_width_m = [1.75, 1.75, 1.75, 1.75]  # standard-ish CARLA lane
    log = make_log(steer=steer, lateral_offset_m=lateral_offset_m, lane_half_width_m=lane_half_width_m)
    m = evaluate(log)
    assert m.max_lateral_offset == 3.0
    assert m.off_lane_frac == 0.5  # ticks 1 and 3 exceed 1.75


def test_deviation_fields_are_none_when_not_present_in_log():
    log = make_log(steer=[0.0, 0.1])
    m = evaluate(log)
    assert m.max_lateral_offset is None
    assert m.off_lane_frac is None
    assert m.min_obstacle_clearance is None


# ---------------------------------------------------------------------------
# time-to-collision / completion / braking
# ---------------------------------------------------------------------------

def test_collision_sets_time_to_collision_and_not_completed():
    log = make_log(steer=[0.0] * 10, collision_at=6, termination_reason="collision")
    m = evaluate(log)
    assert m.collided is True
    assert m.time_to_collision_s == 6 * DT_S
    assert m.completed is False


def test_route_completed_with_no_collision():
    log = make_log(steer=[0.0] * 5, termination_reason="route_completed")
    m = evaluate(log)
    assert m.collided is False
    assert m.time_to_collision_s is None
    assert m.completed is True


def test_braking_severity_fields():
    brake = [0.0, 0.0, 0.9, 1.0]  # abrupt spike at tick 2
    log = make_log(steer=[0.0] * 4, brake=brake)
    m = evaluate(log)
    assert m.max_brake == 1.0
    assert m.mean_brake == sum(brake) / 4
    assert m.max_brake_rate > 0


# ---------------------------------------------------------------------------
# severity_score
# ---------------------------------------------------------------------------

def test_severity_score_zero_for_a_perfectly_clean_baseline():
    steer = [0.0] * 10
    brake = [0.0] * 10
    lateral_offset_m = [0.0] * 10
    lane_half_width_m = [1.75] * 10
    nearest_actor_distance_m = [20.0] * 10
    log = make_log(
        steer=steer, brake=brake,
        lateral_offset_m=lateral_offset_m, lane_half_width_m=lane_half_width_m,
        nearest_actor_distance_m=nearest_actor_distance_m,
    )
    m = evaluate(log)
    assert m.severity_score == 0.0


def test_severity_score_dominated_by_collision_term():
    steer = [0.0] * 5
    log_no_collision = make_log(steer=steer, termination_reason="route_completed")
    log_collision = make_log(steer=steer, collision_at=2, termination_reason="collision")
    assert evaluate(log_collision).severity_score - evaluate(log_no_collision).severity_score == 40.0


def test_severity_score_never_exceeds_100():
    n = 50
    steer = [0.9 * math.sin(2 * math.pi * 2.0 * i * DT_S) for i in range(n)]
    lateral_offset_m = [10.0] * n
    lane_half_width_m = [1.75] * n
    nearest_actor_distance_m = [0.0] * n
    log = make_log(
        steer=steer, collision_at=5,
        lateral_offset_m=lateral_offset_m, lane_half_width_m=lane_half_width_m,
        nearest_actor_distance_m=nearest_actor_distance_m,
    )
    m = evaluate(log)
    assert m.severity_score <= 100.0


def test_to_dict_round_trips_all_fields():
    log = make_log(steer=[0.0, 0.1, 0.2])
    m = evaluate(log)
    d = m.to_dict()
    assert isinstance(d, dict)
    for field in EpisodeMetrics.__dataclass_fields__:
        assert field in d
