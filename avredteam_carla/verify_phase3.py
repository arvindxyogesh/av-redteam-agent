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
import gc
import json
import logging
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("verify_phase3")

# Real finding from this Maui run, not a hypothetical: this script creates
# several LeaderBoard-v0 envs back-to-back *within one process* (baseline,
# then each attack, then N stability-check trials) - a pattern Phase 1/2
# never exercised (their CLI always made exactly one env per process).
# CarlaMultiAgentEnv.close() (carla_multi_agent_env.py) only nils out its
# own self._client/self._tm references; the handler objects it owns
# (_ev_handler, _zv_handler, etc.) still hold direct references to the same
# carla.Client/TrafficManager, so the underlying connection/RPC threads
# aren't necessarily torn down by the time close() returns. The very next
# gym.make() in the same process (creating a second client to the same
# host:port) hung until CARLA's client-side 60s timeout three times in a
# row on this Maui node, and that specific timeout escapes as an *uncaught*
# C++ exception (`terminate called after throwing
# carla::client::TimeoutException`) that aborts the whole process (exit
# 134) - not a catchable Python RuntimeError like the already-known
# intermittent load_world() flakiness from Phase 1/2. No amount of
# Python-side try/except fixes an abort of the process itself. A 8s
# gc.collect()+sleep settle delay (still applied below, cheap insurance)
# was NOT enough to fix it by itself - confirmed by still hitting the exact
# same crash with it in place. The actual fix: baseline and each attack
# below now each run in their own freshly-spawned subprocess (self-
# reinvocation via `--_stage`), since a process exit trivially and
# completely tears down every socket/thread/GPU context - the same
# process-per-episode shape Phase 1/2's CLI always used and which never hit
# this bug. The repeated-call stability check deliberately keeps its N
# calls genuinely in-process, unlike baseline/attacks - that in-process
# repetition is exactly what it exists to test for Phase 4's future search
# loops, so isolating it into subprocesses would make it pass trivially and
# defeat its purpose.
SETTLE_SLEEP_S = 8.0
STAGE_SUBPROCESS_RETRIES = 3


def _settle_after_env_close() -> None:
    gc.collect()
    time.sleep(SETTLE_SLEEP_S)


def _unlink_if_exists(path: Path) -> None:
    # Path.unlink(missing_ok=True) is Python 3.8+ only; this env is 3.7.
    if path.exists():
        path.unlink()


def _run_stage_in_subprocess(stage: str, cli_args: argparse.Namespace) -> dict:
    """Runs one stage (baseline or a named attack) in a fresh subprocess by
    re-invoking this same module with --_stage, and returns that stage's
    result dict (EpisodeMetrics.to_dict(), or Trial.to_dict() for an
    attack). Retries a bounded number of times on failure (covers both the
    ordinary intermittent load_world() timeout and, since each attempt is
    its own fresh process, the cross-episode issue this function exists to
    avoid can't recur within a single attempt either)."""
    last_error = None
    for attempt in range(1, STAGE_SUBPROCESS_RETRIES + 1):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            stage_out = Path(f.name)
        cmd = [
            sys.executable, "-m", "avredteam_carla.verify_phase3",
            "--roach-root", cli_args.roach_root,
            "--host", cli_args.host,
            "--port", str(cli_args.port),
            "--carla-map", cli_args.carla_map,
            "--weather-group", cli_args.weather_group,
            "--route-id", str(cli_args.route_id),
            "--out", str(stage_out),
            "--_stage", stage,
        ]
        log.info("[%s attempt %d/%d] spawning subprocess: %s", stage, attempt, STAGE_SUBPROCESS_RETRIES, " ".join(cmd))
        proc = subprocess.run(cmd)
        if proc.returncode == 0 and stage_out.exists():
            try:
                return json.loads(stage_out.read_text())
            finally:
                _unlink_if_exists(stage_out)
        last_error = f"subprocess for stage {stage!r} exited {proc.returncode}"
        if attempt < STAGE_SUBPROCESS_RETRIES:
            log.warning("%s (attempt %d/%d) - retrying", last_error, attempt, STAGE_SUBPROCESS_RETRIES)
        else:
            log.warning("%s - out of retries", last_error)
        _unlink_if_exists(stage_out)
        _settle_after_env_close()
    raise RuntimeError(last_error)

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
    memory growth across repeats.

    Writes each call's result to log_dir/phase3_stability_partial.json as it
    goes, not just at the end: this loop runs genuinely in-process (that's
    the point - see docs/evaluator.md's stability-check section), and a
    real run on Maui found it can abort the whole process outright partway
    through. Without incremental writes, a crash on e.g. call 2 would
    discard call 1's already-good data along with it.
    """
    from avredteam_carla.runner import run_trial

    partial_path = Path(log_dir) / "phase3_stability_partial.json"

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
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(json.dumps({"n_calls_completed": i + 1, "calls": results}, indent=2, default=str))
        _settle_after_env_close()

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
    # Hidden worker-mode flag: run_stage_in_subprocess() re-invokes this same
    # module with this set to run exactly one stage and exit - see the
    # module-level comment on SETTLE_SLEEP_S for why baseline/attacks each
    # need their own fresh process.
    p.add_argument("--_stage", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    from avredteam_carla.runner import ScenarioConfig, run_baseline, run_trial
    from avredteam_carla.evaluator import EpisodeMetrics
    from avredteam_carla.agents.campaign import Trial

    scenario = ScenarioConfig(
        name=f"{args.carla_map}_{args.weather_group}_route{args.route_id}",
        roach_root=args.roach_root,
        host=args.host,
        port=args.port,
        carla_map=args.carla_map,
        weather_group=args.weather_group,
        route_id=args.route_id,
    )

    if args._stage:
        # Worker mode: run exactly one stage, write its dict, exit. No
        # table printing, no stability check - the parent process handles
        # all of that after collecting every stage's subprocess output.
        if args._stage == "baseline":
            stage_result = run_baseline(scenario).to_dict()
        else:
            attack_params = dict(ATTACK_RUNS)[args._stage]
            stage_result = run_trial(scenario, args._stage, attack_params).to_dict()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(stage_result, indent=2, default=str))
        return 0

    log.info("Running baseline (in a fresh subprocess)...")
    baseline_dict = _run_stage_in_subprocess("baseline", args)
    baseline_metrics = EpisodeMetrics(**baseline_dict)
    log.info("Baseline: %s", baseline_dict)

    trials = []
    for attack_name, attack_params in ATTACK_RUNS:
        log.info("Running attack %s params=%s (in a fresh subprocess)...", attack_name, attack_params)
        trial_dict = _run_stage_in_subprocess(attack_name, args)
        trial = Trial(
            scenario_name=trial_dict["scenario_name"],
            attack_name=trial_dict["attack_name"],
            attack_params=trial_dict["attack_params"],
            metrics=EpisodeMetrics(**trial_dict["metrics"]),
        )
        log.info("%s: %s", attack_name, trial_dict["metrics"])
        trials.append(trial)

    def _write_result(stability_value) -> dict:
        result = {
            "scenario": scenario.name,
            "baseline_metrics": baseline_metrics.to_dict(),
            "trials": [t.to_dict() for t in trials],
            "stability_check": stability_value,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str))
        return result

    # Written *before* the stability check, not just after: that check
    # deliberately runs several envs genuinely in-process (see below) and
    # can abort the whole process outright - confirmed on this Maui run,
    # not a hypothetical. Without this, that abort would silently discard
    # the baseline + all three attacks' already-good results along with it,
    # since nothing had been written to --out yet.
    _write_result(stability_value=None)
    log.info("Wrote baseline+attack results to %s (stability check not yet run)", args.out)

    stability = None
    if not args.skip_stability:
        log.info("Running repeated-call stability check (%d calls)...", args.stability_calls)
        stability = run_stability_check(scenario, args.stability_calls, Path(args.out).parent)

    result = _write_result(stability_value=stability)
    log.info("Wrote verification results to %s", args.out)

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
