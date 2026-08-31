"""Phase 4 Step 6 verification: runs all three search methods (random
search, Bayesian optimization, LLM agent) against the existing Town01
scenario with a small shared budget and a fixed seed, through the real
subprocess-per-trial infrastructure (Step 5), and prints the acceptance
table with real numbers.

This is NOT the real experiment (that's Phase 6's full budget x seed x
scenario sweep) - it's a smoke test proving all three methods work
end-to-end through the real infrastructure, including at least one
observed retry-recovery if the node is under any load during the run
(docs/evaluator.md's Phase 3 finding: this node's load is bursty).

Needs a live CARLA server, `pip install optuna anthropic` in the
carla-redteam env, and ANTHROPIC_API_KEY set for the LLM agent - not
runnable in a dev sandbox without all three. See docs/search_methods.md
for what could/couldn't be verified without them.

Usage:
    python -m avredteam_carla.verify_phase4 \\
        --roach-root /data/savyo/carla-redteam/roach \\
        --host localhost --port 2100 \\
        --budget 10 --search-seed 2021 \\
        --sanity-frames-root /data/savyo/carla-redteam/phase4_verification/frames \\
        --out logs/phase4_verification.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("verify_phase4")


def _require_under_data_dir(path: str, label: str) -> None:
    """Step 5's acceptance criterion: "confirm every output path resolves
    under /data/savyo, don't just assume." Checked explicitly here rather
    than only in a doc comment - a relative path (this repo checkout lives
    under /home, quota-limited per TASK.md's original Phase 1 decision)
    would silently violate this, so fail loudly instead."""
    resolved = str(Path(path).expanduser().resolve())
    if not resolved.startswith("/data/"):
        raise ValueError(
            f"{label}={path!r} resolves to {resolved!r}, which is not under /data/$USER "
            f"(TASK.md's directory-layout decision - /home is quota-limited on Maui). "
            f"Pass an absolute path under /data explicitly."
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roach-root", required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--carla-map", default="Town01")
    p.add_argument("--weather-group", default="simple")
    p.add_argument("--route-id", type=int, default=0)
    p.add_argument("--budget", type=int, default=10, help="Trials per method - small and shared, per Step 6 (this is a smoke test, not Phase 6's real sweep)")
    p.add_argument("--search-seed", type=int, default=2021)
    p.add_argument("--sanity-frames-root", required=True, help="Must resolve under /data/$USER - see Step 5")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    _require_under_data_dir(args.sanity_frames_root, "--sanity-frames-root")
    _require_under_data_dir(args.out, "--out")

    from avredteam_carla.runner import ScenarioConfig, run_baseline
    from avredteam_carla.preflight import preflight_snapshot
    from avredteam_carla.agents.isolated_runner import run_trial_isolated
    from avredteam_carla.agents.random_search import RandomSearch
    from avredteam_carla.agents.bayesian_search import BayesianSearch
    from avredteam_carla.agents.llm_agent_search import LLMAgentSearch
    from avredteam_carla.agents.campaign import TRIAL_OUTCOME_INFRA_FAILURE

    scenario = ScenarioConfig(
        name=f"{args.carla_map}_{args.weather_group}_route{args.route_id}",
        roach_root=args.roach_root,
        host=args.host,
        port=args.port,
        carla_map=args.carla_map,
        weather_group=args.weather_group,
        route_id=args.route_id,
    )

    log.info("Pre-flight snapshot...")
    preflight = preflight_snapshot()
    log.info("Preflight: disk=%s load=%s picked_gpu=%s", preflight["disk"], preflight["load_average"], preflight["picked_gpu"])

    # One baseline run, shared across all three methods' campaigns (Step 1's
    # SearchMethod.run_campaign() docstring: this is exactly why
    # baseline_metrics is fetched once by the caller, not per method). This
    # is a single in-process run_episode() call, not a repeated one - Phase
    # 3's stability finding was specifically about *repeated* in-process
    # calls, which this verification script never does (every trial below
    # goes through run_trial_isolated's subprocess).
    log.info("Running baseline...")
    baseline_metrics = run_baseline(scenario)
    log.info("Baseline: severity_score=%.1f", baseline_metrics.severity_score)

    methods = [RandomSearch(), BayesianSearch(), LLMAgentSearch()]
    results = {}

    for method in methods:
        trial_index = {"n": 0}

        def run_trial(scenario, attack_name, attack_params, baseline_severity, _method=method, _idx=trial_index):
            frames_dir = f"{args.sanity_frames_root}/{_method.name}/trial_{_idx['n']:03d}"
            _idx["n"] += 1
            return run_trial_isolated(
                scenario, attack_name, attack_params, baseline_severity, sanity_frames_dir=frames_dir,
            )

        log.info("Running %s (budget=%d, seed=%d)...", method.name, args.budget, args.search_seed)
        t0 = time.time()
        campaign = method.run_campaign(
            scenario, budget=args.budget, seed=args.search_seed, run_trial=run_trial, baseline_metrics=baseline_metrics,
        )
        campaign.preflight = preflight
        elapsed_s = time.time() - t0

        n_infra_failures = len(campaign.infra_failure_trials())
        best = campaign.best_trial_by_delta()
        results[method.name] = {
            "campaign": campaign.to_dict(),
            "elapsed_s": elapsed_s,
            "n_trials": len(campaign.trials),
            "n_infra_failures": n_infra_failures,
            "best_delta_severity": best.delta_severity if best else None,
        }
        log.info(
            "%s done in %.1fs: %d trials, %d infra_failures, best delta_severity=%s",
            method.name, elapsed_s, len(campaign.trials), n_infra_failures,
            f"{best.delta_severity:.2f}" if best else "n/a",
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "scenario": scenario.name,
        "budget": args.budget,
        "search_seed": args.search_seed,
        "preflight": preflight,
        "baseline_metrics": baseline_metrics.to_dict(),
        "results": results,
    }, indent=2, default=str))
    log.info("Wrote verification results to %s", out_path)

    print()
    print("| Method | Trials completed | Infra failures | Best delta_severity | Wall-clock time |")
    print("|---|---|---|---|---|")
    for name, r in results.items():
        best_str = f"{r['best_delta_severity']:.1f}" if r["best_delta_severity"] is not None else "n/a"
        print(f"| {name} | {r['n_trials'] - r['n_infra_failures']}/{r['n_trials']} | {r['n_infra_failures']} | {best_str} | {r['elapsed_s']:.1f}s |")

    print()
    print("Example agent_notes (first 3 per method):")
    for name, r in results.items():
        print(f"  {name}:")
        for note in r["campaign"]["agent_notes"][:3]:
            print(f"    - {note}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
