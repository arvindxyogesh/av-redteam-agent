"""Random search - the floor baseline every other Phase 4 method is
compared against (docs/search_methods.md Step 2). Deliberately as simple
as SearchMethod allows: uniform sampling over each candidate attack's
TunableParam ranges, one attack type picked uniformly per trial (or the
scenario's fixed_attack_name, if set - see search.attack_pool()). No
adaptation, no memory of past trials' results.
"""
from __future__ import annotations

import random

from avredteam_carla.agents.campaign import CampaignResult
from avredteam_carla.agents.search import SearchMethod, TrialRunner, attack_pool, sample_uniform_params
from avredteam_carla.evaluator import EpisodeMetrics
from avredteam_carla.runner import ScenarioConfig


class RandomSearch(SearchMethod):
    name = "random_search"

    def run_campaign(
        self,
        scenario: ScenarioConfig,
        budget: int,
        seed: int,
        run_trial: TrialRunner,
        baseline_metrics: EpisodeMetrics,
    ) -> CampaignResult:
        rng = random.Random(seed)
        pool = attack_pool(scenario)
        attack_names = sorted(pool)  # deterministic iteration order for a fixed seed

        campaign = CampaignResult(scenario_name=scenario.name, baseline_metrics=baseline_metrics)

        for i in range(budget):
            attack_name = rng.choice(attack_names)
            params = sample_uniform_params(rng, pool[attack_name].tunable_params)

            trial = run_trial(scenario, attack_name, params, baseline_metrics.severity_score)
            campaign.add_trial(trial)
            campaign.agent_notes.append(
                f"trial {i}: uniform random sample, attack={attack_name} params={params}"
            )

        return campaign
