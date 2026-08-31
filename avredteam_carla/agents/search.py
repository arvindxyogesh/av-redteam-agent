"""Shared interface every Phase 4 search method (random search, Bayesian
optimization, LLM agent) implements - see docs/search_methods.md Step 1.

A SearchMethod's job is exactly: given a scenario and a trial budget,
decide which attack + params to try next, run it, and repeat until budget
is exhausted, recording a per-trial rationale note as it goes. Nothing
about how a trial is actually *run* (subprocess isolation, retry-with-
backoff, TrialOutcome bookkeeping) belongs here - that's
avredteam_carla.agents.isolated_runner's job, injected into every
SearchMethod as a plain callable so this module (and random_search.py/
bayesian_search.py) stay fully unit-testable without a live CARLA server.
"""
from __future__ import annotations

from typing import Callable, Optional

from avredteam_carla.agents.campaign import CampaignResult, Trial
from avredteam_carla.attacks.base import TunableParam
from avredteam_carla.evaluator import EpisodeMetrics
from avredteam_carla.runner import ScenarioConfig

# A TrialRunner runs exactly one attack_name/attack_params combination
# against a scenario, given the scenario's baseline severity_score (so the
# returned Trial carries delta_severity), and returns a populated Trial -
# outcome=success or outcome=infra_failure, never raises for an ordinary
# trial failure (only for a genuine caller/programming error). Phase 4's
# real campaigns pass avredteam_carla.agents.isolated_runner.run_trial_isolated;
# unit tests pass a small stub that never touches CARLA. Matches
# avredteam_carla.runner.run_trial's real signature.
TrialRunner = Callable[[ScenarioConfig, str, dict, float], Trial]


class SearchMethod:
    """Base class for a search method.

    Every method must, per docs/search_methods.md Step 1:
      - run exactly `budget` trials (a trial that infra-fails still
        consumes one slot of budget rather than being silently replaced -
        see docs/search_methods.md "Failed-trial policy" for why),
      - be reproducible for a fixed `seed`,
      - populate CampaignResult.agent_notes with one rationale string per
        trial, in the same order trials were run (even "trial N: uniform
        random sample" is a valid note - this keeps later analysis code
        uniform across methods rather than special-casing the LLM agent as
        the only one with notes worth reading).
    """

    name: str = "base"

    def run_campaign(
        self,
        scenario: ScenarioConfig,
        budget: int,
        seed: int,
        run_trial: TrialRunner,
        baseline_metrics: EpisodeMetrics,
    ) -> CampaignResult:
        """baseline_metrics is fetched once by the caller (run_baseline(),
        or a synthetic EpisodeMetrics in a unit test) and passed in rather
        than fetched here, for two reasons: it keeps SearchMethod testable
        without CARLA, and it lets a top-level runner comparing all three
        methods on the same scenario (Phase 4's Step 6 verification) share
        one baseline run across all three campaigns instead of re-running
        it three times."""
        raise NotImplementedError


def attack_pool(scenario: ScenarioConfig) -> dict:
    """The attack name -> TunableParam-schema mapping a search method may
    choose from for this scenario: every registered attack, or just the
    one scenario.fixed_attack_name pins (docs/search_methods.md Step 2's
    "fixed attack type if scenario config specifies one - support both").
    Reads avredteam_carla.attacks.registry.ATTACK_REGISTRY programmatically
    rather than hand-duplicating attack names, so a future 4th attack type
    needs no change here.
    """
    from avredteam_carla.attacks.registry import ATTACK_REGISTRY

    if scenario.fixed_attack_name is not None:
        if scenario.fixed_attack_name not in ATTACK_REGISTRY:
            raise ValueError(
                f"scenario.fixed_attack_name={scenario.fixed_attack_name!r} not in "
                f"ATTACK_REGISTRY: {sorted(ATTACK_REGISTRY)}"
            )
        return {scenario.fixed_attack_name: ATTACK_REGISTRY[scenario.fixed_attack_name]}
    return dict(ATTACK_REGISTRY)


def sample_uniform_params(rng, tunable_params: tuple) -> dict:
    """Uniformly samples one value per TunableParam in its declared
    [low, high] range (bool params: a coin flip). Shared by random search
    (every trial) and by the LLM agent's system-prompt-building code (to
    describe each attack's range without hand-written prose - see
    docs/search_methods.md Step 4).
    """
    params = {}
    for p in tunable_params:
        if p.type == "bool":
            params[p.name] = rng.random() < 0.5
        elif p.type == "int":
            params[p.name] = rng.randint(int(p.low), int(p.high))
        else:  # "float"
            params[p.name] = rng.uniform(p.low, p.high)
    return params


def tunable_params_for(attack_cls) -> tuple:
    """TunableParam schema for a registered attack class - a thin,
    named accessor so callers don't reach into `.tunable_params` directly
    and so this is the one place that would need to change if that
    attribute were ever renamed."""
    return attack_cls.tunable_params
