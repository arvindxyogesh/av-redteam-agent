"""Trial/CampaignResult data model - the shape every Phase 4 search method
(random search, Bayesian opt, LLM agent) and later analysis code shares, so
none of them need to special-case CARLA/Roach specifics. Kept deliberately
minimal and stable: field names and types here should not change once
Phase 4 starts depending on them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from avredteam_carla.evaluator import EpisodeMetrics

# Phase 4 decision (docs/search_methods.md "Failed-trial policy"): a trial
# that exhausts its subprocess retries is a distinct outcome, never a Trial
# with severity_score=0 - that would silently pollute severity comparisons
# with a value that means "infrastructure died," not "nothing happened."
TRIAL_OUTCOME_SUCCESS = "success"
TRIAL_OUTCOME_INFRA_FAILURE = "infra_failure"


@dataclass(frozen=True)
class Trial:
    scenario_name: str
    attack_name: str
    attack_params: dict
    # None only for outcome=TRIAL_OUTCOME_INFRA_FAILURE - an infra-failed
    # trial never produced a completed episode to evaluate.
    metrics: Optional[EpisodeMetrics]
    outcome: str = TRIAL_OUTCOME_SUCCESS
    # Phase 4 decision (docs/search_methods.md "Optimize against delta
    # severity, not raw"): every search method's actual objective is
    # metrics.severity_score - baseline_severity for the scenario in use.
    # Stored directly on the Trial (not left to be recomputed by a reader)
    # so a JSON-logged campaign is self-contained.
    baseline_severity: Optional[float] = None
    error: Optional[str] = None  # last subprocess error, only set on infra_failure

    def __post_init__(self):
        if self.outcome not in (TRIAL_OUTCOME_SUCCESS, TRIAL_OUTCOME_INFRA_FAILURE):
            raise ValueError(f"Trial: unknown outcome {self.outcome!r}")
        if self.outcome == TRIAL_OUTCOME_SUCCESS and self.metrics is None:
            raise ValueError("Trial: outcome=success requires metrics")
        if self.outcome == TRIAL_OUTCOME_INFRA_FAILURE and self.metrics is not None:
            raise ValueError("Trial: outcome=infra_failure must not carry metrics")

    @property
    def delta_severity(self) -> Optional[float]:
        if self.metrics is None or self.baseline_severity is None:
            return None
        return self.metrics.severity_score - self.baseline_severity

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "attack_name": self.attack_name,
            "attack_params": dict(self.attack_params),
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
            "outcome": self.outcome,
            "baseline_severity": self.baseline_severity,
            "delta_severity": self.delta_severity,
            "error": self.error,
        }


@dataclass
class CampaignResult:
    scenario_name: str
    baseline_metrics: EpisodeMetrics
    trials: list = field(default_factory=list)  # list[Trial]
    agent_notes: list = field(default_factory=list)  # list[str]
    # Phase 4 decision (docs/search_methods.md "Pre-flight check"): the
    # disk/GPU/load snapshot taken once before a campaign starts, so later
    # analysis can check whether node conditions correlated with anything
    # odd in this campaign's results. None for a CampaignResult built
    # without a preflight check (e.g. in unit tests).
    preflight: Optional[dict] = None

    def add_trial(self, trial: Trial) -> None:
        if trial.scenario_name != self.scenario_name:
            raise ValueError(
                f"Trial scenario_name {trial.scenario_name!r} doesn't match "
                f"CampaignResult scenario_name {self.scenario_name!r}"
            )
        self.trials.append(trial)

    def successful_trials(self) -> list:
        return [t for t in self.trials if t.outcome == TRIAL_OUTCOME_SUCCESS]

    def infra_failure_trials(self) -> list:
        return [t for t in self.trials if t.outcome == TRIAL_OUTCOME_INFRA_FAILURE]

    def sorted_by_severity(self, descending: bool = True) -> list:
        """Successful trials ordered by metrics.severity_score. Highest-
        severity first by default - the typical "what's the worst thing
        found so far" view a search method or a human reviewing results
        would want. infra_failure trials have no metrics and are excluded,
        not sorted as if severity_score=0 (see TRIAL_OUTCOME_INFRA_FAILURE)."""
        return sorted(
            self.successful_trials(), key=lambda t: t.metrics.severity_score, reverse=descending
        )

    def sorted_by_delta_severity(self, descending: bool = True) -> list:
        """Same as sorted_by_severity(), but by delta_severity - the actual
        quantity every Phase 4 search method optimizes (docs/search_methods.md
        "Optimize against delta severity, not raw")."""
        return sorted(
            self.successful_trials(),
            key=lambda t: t.delta_severity,
            reverse=descending,
        )

    def best_trial(self) -> Optional[Trial]:
        ranked = self.sorted_by_severity()
        return ranked[0] if ranked else None

    def best_trial_by_delta(self) -> Optional[Trial]:
        ranked = self.sorted_by_delta_severity()
        return ranked[0] if ranked else None

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "trials": [t.to_dict() for t in self.trials],
            "agent_notes": list(self.agent_notes),
            "preflight": self.preflight,
        }
