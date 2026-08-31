from avredteam_carla.compare_episodes import sign_flip_count, episode_metrics, paired_control_deviation, compare


def test_sign_flip_count_detects_oscillation():
    # steer: up, down, up, down -> 3 direction reversals
    assert sign_flip_count([0.0, 0.5, 0.0, 0.5, 0.0]) == 3


def test_sign_flip_count_ignores_flat_runs():
    # a flat spot (no change) shouldn't itself count as a reversal
    assert sign_flip_count([0.0, 0.5, 0.5, 0.5, 0.0]) == 1


def test_sign_flip_count_monotonic_series_has_no_flips():
    assert sign_flip_count([0.0, 0.1, 0.2, 0.3]) == 0


def _fake_log(steers, brakes, speeds, termination="route_completed"):
    ticks = [
        {"tick": i, "steer": s, "throttle": 0.5, "brake": b, "ground_truth_speed": v}
        for i, (s, b, v) in enumerate(zip(steers, brakes, speeds))
    ]
    return {"meta": {"termination_reason": termination}, "ticks": ticks}


def test_episode_metrics_basic_stats():
    log = _fake_log(steers=[0.0, 0.2, -0.1], brakes=[0.0, 0.5, 1.0], speeds=[10.0, 5.0, 0.0])
    m = episode_metrics(log)
    assert m["n_ticks"] == 3
    assert m["mean_brake"] == 0.5
    assert m["max_brake"] == 1.0
    assert m["mean_speed"] == 5.0
    assert m["min_speed"] == 0.0
    assert m["termination_reason"] == "route_completed"


def test_paired_control_deviation_matches_lengths():
    clean = _fake_log(steers=[0.0, 0.0, 0.0], brakes=[0.0, 0.0, 0.0], speeds=[10, 10, 10])
    attacked = _fake_log(steers=[0.0, 0.5, -0.5], brakes=[0.0, 1.0, 0.0], speeds=[10, 2, 8])
    dev = paired_control_deviation(clean["ticks"], attacked["ticks"])
    assert dev["n_compared"] == 3
    assert dev["mean_abs_steer_diff"] == (0.0 + 0.5 + 0.5) / 3
    assert dev["max_abs_brake_diff"] == 1.0


def test_paired_control_deviation_handles_different_lengths():
    clean = _fake_log(steers=[0.0] * 5, brakes=[0.0] * 5, speeds=[10] * 5)
    attacked = _fake_log(steers=[0.0, 1.0], brakes=[0.0, 1.0], speeds=[10, 0])  # attack ended the episode early
    dev = paired_control_deviation(clean["ticks"], attacked["ticks"])
    assert dev["n_compared"] == 2  # only compares the overlapping prefix


def test_compare_bundles_everything():
    clean = _fake_log(steers=[0.0, 0.0], brakes=[0.0, 0.0], speeds=[10, 10])
    attacked = _fake_log(steers=[0.0, 0.3], brakes=[0.0, 1.0], speeds=[10, 1])
    attacked["meta"]["attack"] = {"name": "phantom_actor", "params": {"distance_m": 10.0}}
    result = compare(clean, attacked)
    assert result["attack_meta"]["name"] == "phantom_actor"
    assert result["paired_control_deviation"]["n_compared"] == 2
