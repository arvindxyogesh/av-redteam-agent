"""Unit tests for BayesianSearch (docs/search_methods.md Step 3). Pure
Python + optuna, no CARLA - the TrialRunner is stubbed exactly like
tests/test_search.py's random-search tests."""
import optuna
import pytest

from avredteam_carla.agents.bayesian_search import BayesianSearch, INFRA_FAILURE_OBJECTIVE
from avredteam_carla.agents.campaign import Trial, TRIAL_OUTCOME_INFRA_FAILURE
from avredteam_carla.evaluator import evaluate
from avredteam_carla.runner import ScenarioConfig
from tests.test_evaluator import make_log

BASELINE_METRICS = evaluate(make_log(steer=[0.0] * 5, termination_reason="route_completed"))


def _scenario(**kwargs):
    return ScenarioConfig(name="s", roach_root="/fake", **kwargs)


def _stub_run_trial(calls, severities_by_attack=None):
    """Deterministic-severity stub: geometry_spoof's severity scales with
    its own max_offset_m param, everything else is flat - gives the
    sampler something real to (in principle) exploit, for the
    "runs without error and improves nothing weird" smoke test below."""

    def run_trial(scenario, attack_name, attack_params, baseline_severity):
        calls.append((attack_name, dict(attack_params)))
        collision_at = 0 if attack_name == "geometry_spoof" and attack_params.get("max_offset_m", 0) > 5 else None
        m = evaluate(make_log(
            steer=[0.0] * 5,
            collision_at=collision_at,
            termination_reason="collision" if collision_at is not None else "route_completed",
        ))
        return Trial(scenario.name, attack_name, attack_params, m, baseline_severity=baseline_severity)

    return run_trial


def test_bayesian_search_runs_exactly_budget_trials():
    calls = []
    campaign = BayesianSearch().run_campaign(
        _scenario(), budget=6, seed=0, run_trial=_stub_run_trial(calls), baseline_metrics=BASELINE_METRICS
    )
    assert len(calls) == 6
    assert len(campaign.trials) == 6
    assert len(campaign.agent_notes) == 6


def test_bayesian_search_reproducible_for_fixed_seed():
    calls_a, calls_b = [], []
    BayesianSearch().run_campaign(_scenario(), budget=5, seed=7, run_trial=_stub_run_trial(calls_a), baseline_metrics=BASELINE_METRICS)
    BayesianSearch().run_campaign(_scenario(), budget=5, seed=7, run_trial=_stub_run_trial(calls_b), baseline_metrics=BASELINE_METRICS)
    assert calls_a == calls_b


def test_bayesian_search_honors_fixed_attack_name():
    calls = []
    BayesianSearch().run_campaign(
        _scenario(fixed_attack_name="phantom_actor"),
        budget=5, seed=0, run_trial=_stub_run_trial(calls), baseline_metrics=BASELINE_METRICS,
    )
    assert all(name == "phantom_actor" for name, _ in calls)


def test_bayesian_search_params_within_declared_ranges():
    calls = []
    BayesianSearch().run_campaign(
        _scenario(fixed_attack_name="geometry_spoof"),
        budget=15, seed=0, run_trial=_stub_run_trial(calls), baseline_metrics=BASELINE_METRICS,
    )
    for _, params in calls:
        assert -10.0 <= params["max_offset_m"] <= 10.0
        assert 1 <= params["ramp_ticks"] <= 300


def test_bayesian_search_infra_failure_scored_and_recorded():
    def flaky_run_trial(scenario, attack_name, attack_params, baseline_severity):
        return Trial(scenario.name, attack_name, attack_params, None, outcome=TRIAL_OUTCOME_INFRA_FAILURE, error="timeout")

    campaign = BayesianSearch().run_campaign(
        _scenario(), budget=3, seed=0, run_trial=flaky_run_trial, baseline_metrics=BASELINE_METRICS
    )
    assert len(campaign.trials) == 3
    assert all(t.outcome == TRIAL_OUTCOME_INFRA_FAILURE for t in campaign.trials)
    assert campaign.successful_trials() == []
    assert all("infra_failure" in note for note in campaign.agent_notes)
