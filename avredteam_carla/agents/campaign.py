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


@dataclass(frozen=True)
class Trial:
    scenario_name: str
    attack_name: str
    attack_params: dict
    metrics: EpisodeMetrics

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "attack_name": self.attack_name,
            "attack_params": dict(self.attack_params),
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class CampaignResult:
    scenario_name: str
    baseline_metrics: EpisodeMetrics
    trials: list = field(default_factory=list)  # list[Trial]
    agent_notes: list = field(default_factory=list)  # list[str]

    def add_trial(self, trial: Trial) -> None:
        if trial.scenario_name != self.scenario_name:
            raise ValueError(
                f"Trial scenario_name {trial.scenario_name!r} doesn't match "
                f"CampaignResult scenario_name {self.scenario_name!r}"
            )
        self.trials.append(trial)

    def sorted_by_severity(self, descending: bool = True) -> list:
        """Trials ordered by metrics.severity_score. Highest-severity first
        by default - the typical "what's the worst thing found so far"
        view a search method or a human reviewing results would want."""
        return sorted(
            self.trials, key=lambda t: t.metrics.severity_score, reverse=descending
        )

    def best_trial(self) -> Optional[Trial]:
        ranked = self.sorted_by_severity()
        return ranked[0] if ranked else None

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "trials": [t.to_dict() for t in self.trials],
            "agent_notes": list(self.agent_notes),
        }
