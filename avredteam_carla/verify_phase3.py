"""Phase 3 verification: runs the baseline + all three Phase 2 attacks
through the new evaluator/runner against the Phase 1 Town01 route, checks
run_trial() stability across repeated back-to-back calls, and prints the
acceptance table with real numbers.

Needs a live CARLA server - not runnable in a dev sandbox without CARLA.
See docs/evaluator.md #8 for what could/couldn't be verified without it.

Usage:
    python -m avredteam_carla.verify_phase3 \\
        --roach-root /data/savyo/carla-redteam/roach \\
        --host localhost --port 2100 \\
        --out logs/phase3_verification.json
"""
from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("verify_phase3")

# Same params Phase 2 actually verified against real hardware (docs/attacks.md
# #6) - reusing them here means Phase 3's numeric metrics can be checked
# directly against Phase 2's already-confirmed qualitative outcomes.
ATTACK_RUNS = [
    ("channel_noise", {"channel": 1, "amplitude": 100.0, "frequency_hz": 2.0}),
    ("geometry_spoof", {"max_offset_m": 3.0, "ramp_ticks": 30}),
    ("phantom_actor", {"distance_m": 15.0, "trigger_tick": 50}),
]


def _actor_count(host: str, port: int) -> int:
    import carla

    client = carla.Client(host, port)
    client.set_timeout(30.0)
    return len(client.get_world().get_actors())


def run_stability_check(scenario, n_calls: int, log_dir: Path) -> dict:
    """docs/evaluator.md #8 / Phase 3 brief Step 4: run_trial() back-to-back
    n_calls times, confirm no actor leak (world actor count returns to a
    stable baseline after each call's env.close()) and no runaway timing/
    memory growth across repeats."""
    from avredteam_carla.runner import run_trial

    results = []
    for i in range(n_calls):
        t0 = time.time()
        rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        trial = run_trial(
            scenario, "phantom_actor",
            {"distance_m": 15.0, "trigger_tick": 20},
        )
        elapsed_s = time.time() - t0
        rss_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        actor_count = _actor_count(scenario.host, scenario.port)
        results.append(
            {
                "call": i,
                "elapsed_s": elapsed_s,
                "rss_after_kb": rss_after_kb,
                "rss_delta_kb": rss_after_kb - rss_before_kb,
                "actor_count_after_close": actor_count,
                "n_ticks": trial.metrics.n_ticks,
                "severity_score": trial.metrics.severity_score,
            }
        )
        log.info(
            "Stability call %d/%d: %.1fs, %d ticks, actor_count_after_close=%d, rss=%dkB",
            i + 1, n_calls, elapsed_s, trial.metrics.n_ticks, actor_count, rss_after_kb,
        )

    actor_counts = [r["actor_count_after_close"] for r in results]
    elapsed_times = [r["elapsed_s"] for r in results]
    return {
        "n_calls": n_calls,
        "calls": results,
        "actor_count_stable": (max(actor_counts) - min(actor_counts)) <= 2,  # small tolerance for CARLA's own bookkeeping
        "actor_count_range": [min(actor_counts), max(actor_counts)],
        "timing_stable": (max(elapsed_times) / min(elapsed_times)) < 2.0 if min(elapsed_times) > 0 else False,
        "elapsed_s_range": [min(elapsed_times), max(elapsed_times)],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roach-root", required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--carla-map", default="Town01")
    p.add_argument("--weather-group", default="simple")
    p.add_argument("--route-id", type=int, default=0)
    p.add_argument("--stability-calls", type=int, default=6, help="How many back-to-back run_trial calls for the leak check")
    p.add_argument("--skip-stability", action="store_true", help="Skip the repeated-call stability check")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from avredteam_carla.runner import ScenarioConfig, run_baseline, run_trial

    scenario = ScenarioConfig(
        name=f"{args.carla_map}_{args.weather_group}_route{args.route_id}",
        roach_root=args.roach_root,
        host=args.host,
        port=args.port,
        carla_map=args.carla_map,
        weather_group=args.weather_group,
        route_id=args.route_id,
    )

    log.info("Running baseline...")
    baseline_metrics = run_baseline(scenario)
    log.info("Baseline: %s", baseline_metrics.to_dict())

    trials = []
    for attack_name, attack_params in ATTACK_RUNS:
        log.info("Running attack %s params=%s ...", attack_name, attack_params)
        trial = run_trial(scenario, attack_name, attack_params)
        log.info("%s: %s", attack_name, trial.metrics.to_dict())
        trials.append(trial)

    stability = None
    if not args.skip_stability:
        log.info("Running repeated-call stability check (%d calls)...", args.stability_calls)
        stability = run_stability_check(scenario, args.stability_calls, Path(args.out).parent)

    result = {
        "scenario": scenario.name,
        "baseline_metrics": baseline_metrics.to_dict(),
        "trials": [t.to_dict() for t in trials],
        "stability_check": stability,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    log.info("Wrote verification results to %s", out_path)

    # Print the acceptance table directly, so it can be pasted straight
    # into docs/evaluator.md / the PR description.
    print()
    print("| Condition | Severity | Chattering rate | Max jerk | Time-to-collision (or completed) | Max brake |")
    print("|---|---|---|---|---|---|")

    def row(name, m):
        ttc = f"{m.time_to_collision_s:.1f}s" if m.time_to_collision_s is not None else ("completed" if m.completed else "n/a")
        print(f"| {name} | {m.severity_score:.1f} | {m.chattering_rate:.3f} | {m.max_steering_jerk:.2f} | {ttc} | {m.max_brake:.2f} |")

    row("Baseline (clean)", baseline_metrics)
    for trial in trials:
        row(trial.attack_name, trial.metrics)

    if stability is not None:
        print()
        print(f"Stability check ({stability['n_calls']} back-to-back run_trial calls):")
        print(f"  actor_count range: {stability['actor_count_range']} (stable={stability['actor_count_stable']})")
        print(f"  elapsed_s range: {stability['elapsed_s_range']} (stable={stability['timing_stable']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
