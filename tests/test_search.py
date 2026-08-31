"""Unit tests for the shared SearchMethod interface + random search
(docs/search_methods.md Steps 1-2). Pure Python - the TrialRunner is
stubbed, no CARLA needed. See tests/test_evaluator.py's make_log() for the
same synthetic-log pattern used elsewhere in this repo.
"""
import random

import pytest

from avredteam_carla.agents.campaign import Trial
from avredteam_carla.agents.random_search import RandomSearch
from avredteam_carla.agents.search import attack_pool, sample_uniform_params
from avredteam_carla.attacks.base import TunableParam
from avredteam_carla.attacks.registry import ATTACK_REGISTRY
from avredteam_carla.evaluator import evaluate
from avredteam_carla.runner import ScenarioConfig
from tests.test_evaluator import make_log

BASELINE_METRICS = evaluate(make_log(steer=[0.0] * 5, termination_reason="route_completed"))


def _stub_run_trial(calls):
    """A TrialRunner that never touches CARLA: returns a Trial whose
    severity_score is derived deterministically from the params, so tests
    can assert on delta_severity without any real episode."""

    def run_trial(scenario, attack_name, attack_params, baseline_severity):
        calls.append((attack_name, dict(attack_params)))
        m = evaluate(make_log(steer=[0.0] * 5, termination_reason="route_completed"))
        return Trial(scenario.name, attack_name, attack_params, m, baseline_severity=baseline_severity)

    return run_trial


def _scenario(**kwargs):
    return ScenarioConfig(name="s", roach_root="/fake", **kwargs)


# --- attack_pool -------------------------------------------------------


def test_attack_pool_returns_full_registry_by_default():
    pool = attack_pool(_scenario())
    assert set(pool) == set(ATTACK_REGISTRY)


def test_attack_pool_respects_fixed_attack_name():
    pool = attack_pool(_scenario(fixed_attack_name="phantom_actor"))
    assert set(pool) == {"phantom_actor"}


def test_attack_pool_rejects_unknown_fixed_attack_name():
    with pytest.raises(ValueError):
        attack_pool(_scenario(fixed_attack_name="not_a_real_attack"))


# --- sample_uniform_params ----------------------------------------------


def test_sample_uniform_params_stays_in_range():
    params = (
        TunableParam("f", "float", default=0.0, low=-5.0, high=5.0),
        TunableParam("i", "int", default=0, low=0, high=10),
        TunableParam("b", "bool", default=False),
    )
    rng = random.Random(0)
    for _ in range(200):
        sampled = sample_uniform_params(rng, params)
        assert -5.0 <= sampled["f"] <= 5.0
        assert 0 <= sampled["i"] <= 10
        assert isinstance(sampled["b"], bool)


# --- RandomSearch --------------------------------------------------------


def test_random_search_runs_exactly_budget_trials():
    calls = []
    campaign = RandomSearch().run_campaign(
        _scenario(), budget=7, seed=1, run_trial=_stub_run_trial(calls), baseline_metrics=BASELINE_METRICS
    )
    assert len(calls) == 7
    assert len(campaign.trials) == 7
    assert len(campaign.agent_notes) == 7


def test_random_search_reproducible_for_fixed_seed():
    calls_a, calls_b = [], []
    RandomSearch().run_campaign(_scenario(), budget=5, seed=42, run_trial=_stub_run_trial(calls_a), baseline_metrics=BASELINE_METRICS)
    RandomSearch().run_campaign(_scenario(), budget=5, seed=42, run_trial=_stub_run_trial(calls_b), baseline_metrics=BASELINE_METRICS)
    assert calls_a == calls_b


def test_random_search_different_seeds_diverge():
    calls_a, calls_b = [], []
    RandomSearch().run_campaign(_scenario(), budget=10, seed=1, run_trial=_stub_run_trial(calls_a), baseline_metrics=BASELINE_METRICS)
    RandomSearch().run_campaign(_scenario(), budget=10, seed=2, run_trial=_stub_run_trial(calls_b), baseline_metrics=BASELINE_METRICS)
    assert calls_a != calls_b


def test_random_search_honors_fixed_attack_name():
    calls = []
    RandomSearch().run_campaign(
        _scenario(fixed_attack_name="channel_noise"),
        budget=5, seed=1, run_trial=_stub_run_trial(calls), baseline_metrics=BASELINE_METRICS,
    )
    assert all(name == "channel_noise" for name, _ in calls)


def test_random_search_baseline_metrics_carried_on_campaign():
    campaign = RandomSearch().run_campaign(
        _scenario(), budget=1, seed=1, run_trial=_stub_run_trial([]), baseline_metrics=BASELINE_METRICS
    )
    assert campaign.baseline_metrics is BASELINE_METRICS
    assert campaign.trials[0].baseline_severity == BASELINE_METRICS.severity_score
