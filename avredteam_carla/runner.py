"""run_trial()/run_baseline() - what Phase 4's search methods actually call:
give me a scenario + attack name + params, get back a structured Trial.

Thin wrapper over run_clean_episode.run_episode() + evaluator.evaluate(),
per the Phase 3 brief's Step 4. World/actor cleanup is handled inside
run_episode() itself (env.close() in a finally block, unchanged from
Phase 1/2) - repeated calls are expected (Phase 4/6 will call this
hundreds of times in a loop), see docs/evaluator.md #8 for the explicit
back-to-back stability check run against a live CARLA server.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from avredteam_carla.agents.campaign import Trial
from avredteam_carla.evaluator import EpisodeMetrics, evaluate


@dataclass(frozen=True)
class ScenarioConfig:
    """Minimal scenario description - Phase 5 adds a proper scenario suite;
    for now this is exactly the Town01 route already verified in Phase 1/2,
    parameterized so run_trial/run_baseline don't hardcode it."""

    name: str
    roach_root: str
    host: str = "localhost"
    port: int = 2000
    seed: int = 2021
    wb_run_path: str = "iccv21-roach/trained-models/1929isj0"
    wb_ckpt_step: Optional[str] = None
    carla_map: str = "Town01"
    weather_group: str = "simple"
    route_id: int = 0
    max_steps: int = 6000
    workdir: Optional[str] = None
    # Phase 4 (docs/search_methods.md Step 2): if set, every search method
    # samples params for this one attack type only, instead of also
    # picking an attack type per trial. Phase 5's proper scenario suite may
    # replace this with something richer; kept minimal for now.
    fixed_attack_name: Optional[str] = None


def _run_episode_kwargs(scenario: ScenarioConfig) -> dict:
    return dict(
        roach_root=scenario.roach_root,
        host=scenario.host,
        port=scenario.port,
        seed=scenario.seed,
        wb_run_path=scenario.wb_run_path,
        wb_ckpt_step=scenario.wb_ckpt_step,
        carla_map=scenario.carla_map,
        weather_group=scenario.weather_group,
        route_id=scenario.route_id,
        max_steps=scenario.max_steps,
        workdir=scenario.workdir,
    )


def run_baseline(scenario: ScenarioConfig) -> EpisodeMetrics:
    """No-attack episode -> EpisodeMetrics. Uses run_episode()'s
    unchanged-from-Phase-1 no-attack path (attack_name=None)."""
    from avredteam_carla.run_clean_episode import run_episode

    log_dict = run_episode(**_run_episode_kwargs(scenario), attack_name=None)
    return evaluate(log_dict)


def run_trial(
    scenario: ScenarioConfig,
    attack_name: str,
    attack_params: Optional[dict] = None,
    baseline_severity: Optional[float] = None,
    sanity_frames_dir: Optional[str] = None,
) -> Trial:
    """Runs one attacked episode and returns a populated Trial - the call
    a Phase 4 search method makes in a loop: give it an attack name +
    params, get back structured metrics to decide what to try next.

    baseline_severity (Phase 4): the scenario's baseline severity_score
    (from run_baseline()), so the returned Trial carries delta_severity -
    the quantity every Phase 4 search method actually optimizes against
    (docs/search_methods.md "Optimize against delta severity, not raw").
    A caller running many trials against the same scenario should call
    run_baseline() once and pass its severity_score here on every
    run_trial() call, rather than re-running the baseline per trial.
    Optional (default None) to keep this call byte-for-byte compatible with
    Phase 3 callers (verify_phase3.py) that don't need delta_severity.
    """
    from avredteam_carla.run_clean_episode import run_episode

    log_dict = run_episode(
        **_run_episode_kwargs(scenario),
        attack_name=attack_name,
        attack_params=attack_params or {},
        sanity_frames_dir=sanity_frames_dir,
    )
    metrics = evaluate(log_dict)
    return Trial(
        scenario_name=scenario.name,
        attack_name=attack_name,
        attack_params=dict(attack_params or {}),
        metrics=metrics,
        baseline_severity=baseline_severity,
    )
