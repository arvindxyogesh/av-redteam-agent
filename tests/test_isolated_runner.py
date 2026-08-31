"""Unit tests for run_trial_isolated (docs/search_methods.md Step 5's
subprocess-per-trial isolation + retry-with-backoff). Both subprocess
spawning and sleep() are injected fakes - no real subprocess, no real
multi-second waits, no CARLA."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from avredteam_carla.agents.campaign import TRIAL_OUTCOME_INFRA_FAILURE, TRIAL_OUTCOME_SUCCESS
from avredteam_carla.agents.isolated_runner import run_trial_isolated
from avredteam_carla.evaluator import evaluate
from avredteam_carla.runner import ScenarioConfig
from tests.test_evaluator import make_log

SCENARIO = ScenarioConfig(name="s", roach_root="/fake")
BASELINE_SEVERITY = 16.1


def _out_path_from_cmd(cmd: list) -> Path:
    return Path(cmd[cmd.index("--out") + 1])


def _write_success_trial(out_path: Path) -> None:
    m = evaluate(make_log(steer=[0.0] * 5, termination_reason="route_completed"))
    d = {
        "scenario_name": "s", "attack_name": "channel_noise", "attack_params": {"amplitude": 50.0},
        "metrics": m.to_dict(), "outcome": TRIAL_OUTCOME_SUCCESS, "baseline_severity": BASELINE_SEVERITY,
        "delta_severity": m.severity_score - BASELINE_SEVERITY, "error": None,
    }
    out_path.write_text(json.dumps(d))


def test_succeeds_on_first_attempt_no_sleep_called():
    calls, sleeps = [], []

    def fake_run(cmd, timeout=None):
        calls.append(cmd)
        _write_success_trial(_out_path_from_cmd(cmd))
        return SimpleNamespace(returncode=0)

    trial = run_trial_isolated(
        SCENARIO, "channel_noise", {"amplitude": 50.0}, BASELINE_SEVERITY,
        run_subprocess=fake_run, sleep=lambda s: sleeps.append(s),
    )
    assert trial.outcome == TRIAL_OUTCOME_SUCCESS
    assert trial.metrics is not None
    assert len(calls) == 1
    assert sleeps == []


def test_retries_with_backoff_then_succeeds():
    calls, sleeps = [], []

    def fake_run(cmd, timeout=None):
        calls.append(cmd)
        if len(calls) < 3:
            return SimpleNamespace(returncode=1)  # fails twice
        _write_success_trial(_out_path_from_cmd(cmd))
        return SimpleNamespace(returncode=0)

    trial = run_trial_isolated(
        SCENARIO, "channel_noise", {}, BASELINE_SEVERITY,
        retries=3, base_backoff_s=8.0, backoff_factor=2.0,
        run_subprocess=fake_run, sleep=lambda s: sleeps.append(s),
    )
    assert trial.outcome == TRIAL_OUTCOME_SUCCESS
    assert len(calls) == 3
    assert sleeps == [8.0, 16.0]  # exponential backoff between the 2 failed attempts


def test_exhausts_retries_returns_infra_failure_not_raise():
    calls = []

    def always_fails(cmd, timeout=None):
        calls.append(cmd)
        return SimpleNamespace(returncode=134)  # matches the real C++ abort exit code

    trial = run_trial_isolated(
        SCENARIO, "channel_noise", {}, BASELINE_SEVERITY,
        retries=3, run_subprocess=always_fails, sleep=lambda s: None,
    )
    assert trial.outcome == TRIAL_OUTCOME_INFRA_FAILURE
    assert trial.metrics is None
    assert trial.delta_severity is None
    assert "134" in trial.error
    assert len(calls) == 3  # exactly `retries` attempts, no more


def test_timeout_counts_as_a_failed_attempt():
    calls = []

    def timing_out(cmd, timeout=None):
        calls.append(cmd)
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    trial = run_trial_isolated(
        SCENARIO, "channel_noise", {}, BASELINE_SEVERITY,
        retries=2, subprocess_timeout_s=5.0, run_subprocess=timing_out, sleep=lambda s: None,
    )
    assert trial.outcome == TRIAL_OUTCOME_INFRA_FAILURE
    assert "timed out" in trial.error
    assert len(calls) == 2


def test_backoff_capped_at_max_backoff_s():
    sleeps = []

    def always_fails(cmd, timeout=None):
        return SimpleNamespace(returncode=1)

    run_trial_isolated(
        SCENARIO, "channel_noise", {}, BASELINE_SEVERITY,
        retries=5, base_backoff_s=10.0, backoff_factor=3.0, max_backoff_s=25.0,
        run_subprocess=always_fails, sleep=lambda s: sleeps.append(s),
    )
    assert sleeps == [10.0, 25.0, 25.0, 25.0]  # 10, 30->capped 25, 90->capped 25, 270->capped 25


def test_infra_failure_trial_preserves_attack_name_and_params():
    def always_fails(cmd, timeout=None):
        return SimpleNamespace(returncode=1)

    trial = run_trial_isolated(
        SCENARIO, "geometry_spoof", {"max_offset_m": 3.0}, BASELINE_SEVERITY,
        retries=1, run_subprocess=always_fails, sleep=lambda s: None,
    )
    assert trial.scenario_name == "s"
    assert trial.attack_name == "geometry_spoof"
    assert trial.attack_params == {"max_offset_m": 3.0}
    assert trial.baseline_severity == BASELINE_SEVERITY


def test_sanity_frames_dir_passed_through_to_worker_command():
    captured_cmd = []

    def fake_run(cmd, timeout=None):
        captured_cmd.extend(cmd)
        _write_success_trial(_out_path_from_cmd(cmd))
        return SimpleNamespace(returncode=0)

    run_trial_isolated(
        SCENARIO, "channel_noise", {}, BASELINE_SEVERITY,
        sanity_frames_dir="/data/savyo/frames/trial_0", run_subprocess=fake_run, sleep=lambda s: None,
    )
    assert "--sanity-frames-dir" in captured_cmd
    assert "/data/savyo/frames/trial_0" in captured_cmd


def test_subprocess_launch_exception_treated_as_infra_failure():
    def raises(cmd, timeout=None):
        raise OSError("fork failed")

    trial = run_trial_isolated(
        SCENARIO, "channel_noise", {}, BASELINE_SEVERITY,
        retries=1, run_subprocess=raises, sleep=lambda s: None,
    )
    assert trial.outcome == TRIAL_OUTCOME_INFRA_FAILURE
    assert "fork failed" in trial.error
