"""Bayesian optimization search method, built on Optuna
(docs/search_methods.md Step 3).

Design choice (brief left this open, deciding + documenting here rather
than guessing silently): **one Optuna study per campaign**, with attack
type sampled as a categorical parameter alongside each attack's own
TunableParams, rather than one study per attack type. Reasoning: `budget`
is a single fixed integer for the whole campaign (Step 1's exact-budget
requirement), and running N separate per-attack studies would need
splitting that budget across attacks up front with no way to adaptively
shift trials toward whichever attack type is actually proving effective -
exactly the kind of exploration Bayesian optimization is supposed to buy
over random search. A single study with attack_name as a categorical
choice is also Optuna's own documented pattern for a conditional/mixed
search space (a "define-by-run" objective that only calls suggest_*() for
the branch it actually takes), so this isn't a workaround - it's the
intended way to express "pick a discrete option, then its own sub-space."
"""
from __future__ import annotations

from avredteam_carla.agents.campaign import CampaignResult, TRIAL_OUTCOME_INFRA_FAILURE
from avredteam_carla.agents.search import SearchMethod, TrialRunner, attack_pool, tunable_params_for
from avredteam_carla.attacks.base import TunableParam
from avredteam_carla.evaluator import EpisodeMetrics
from avredteam_carla.runner import ScenarioConfig

# Fed to Optuna's objective for an infra_failure trial instead of a real
# delta_severity (docs/search_methods.md "Failed-trial policy": a failed
# trial consumes budget but was never scored, so it has no real severity to
# report). Must be a real float (Optuna's objective can't return None) and
# clearly worse than any real delta_severity can be (severity_score is
# bounded to [0, 100], so delta_severity is bounded to [-100, 100]) - this
# teaches TPE to steer away from parameter regions that keep infra-failing,
# without ever being mistaken for a genuinely low-severity outcome.
INFRA_FAILURE_OBJECTIVE = -1000.0


def _suggest_param(opt_trial, attack_name: str, p: TunableParam):
    # Namespaced by attack name: two attacks can each declare a param
    # called e.g. "channel" with different (or coincidentally identical)
    # ranges, and Optuna's TPE tracks one distribution per parameter name
    # across the whole study - namespacing keeps those distributions from
    # ever being conflated, even if today's attacks happen not to collide.
    key = f"{attack_name}.{p.name}"
    if p.type == "bool":
        return opt_trial.suggest_categorical(key, [False, True])
    if p.type == "int":
        return opt_trial.suggest_int(key, int(p.low), int(p.high))
    return opt_trial.suggest_float(key, p.low, p.high)


class BayesianSearch(SearchMethod):
    name = "bayesian_search"

    def run_campaign(
        self,
        scenario: ScenarioConfig,
        budget: int,
        seed: int,
        run_trial: TrialRunner,
        baseline_metrics: EpisodeMetrics,
    ) -> CampaignResult:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        pool = attack_pool(scenario)
        attack_names = sorted(pool)
        campaign = CampaignResult(scenario_name=scenario.name, baseline_metrics=baseline_metrics)

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        trial_counter = {"n": 0}

        def objective(opt_trial) -> float:
            attack_name = opt_trial.suggest_categorical("attack_name", attack_names)
            params = {
                p.name: _suggest_param(opt_trial, attack_name, p)
                for p in tunable_params_for(pool[attack_name])
            }

            trial = run_trial(scenario, attack_name, params, baseline_metrics.severity_score)
            campaign.add_trial(trial)

            i = trial_counter["n"]
            trial_counter["n"] += 1
            if trial.outcome == TRIAL_OUTCOME_INFRA_FAILURE:
                campaign.agent_notes.append(
                    f"trial {i}: optuna TPE sample, attack={attack_name} params={params} "
                    f"-> infra_failure ({trial.error}), scored as {INFRA_FAILURE_OBJECTIVE}"
                )
                return INFRA_FAILURE_OBJECTIVE

            campaign.agent_notes.append(
                f"trial {i}: optuna TPE sample, attack={attack_name} params={params} "
                f"-> delta_severity={trial.delta_severity:.2f}"
            )
            return trial.delta_severity

        # study.optimize's objective never raises (run_trial reports
        # infra_failure as a return value, not an exception - Step 5), so
        # every one of these n_trials calls corresponds to exactly one
        # run_trial() call and one Trial added to campaign - `budget` in,
        # `budget` trials out, per Step 1's exact-budget requirement.
        study.optimize(objective, n_trials=budget)

        return campaign
