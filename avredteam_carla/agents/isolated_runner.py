"""Subprocess-per-trial isolation + retry-with-backoff
(docs/search_methods.md Step 5) - the real implementation of what
docs/evaluator.md's Phase 3 stability check found necessary: repeated
in-process run_trial() calls can abort the whole process via an uncaught
C++ exception that no Python try/except can catch, so every trial across
all three Phase 4 search methods runs in its own freshly-spawned
subprocess (avredteam_carla.agents.trial_worker), with a bounded number of
retries and an exponential backoff between attempts - Phase 3's own
finding was that this node's load is bursty, not steadily high, so a
short wait tends to ride out a spike rather than needing an unbounded one.

This module provides run_trial_isolated(), a drop-in TrialRunner
(avredteam_carla.agents.search.TrialRunner) for all three search methods'
run_campaign() calls - the only production TrialRunner Phase 4 ships;
tests use their own stubs (see tests/test_search.py etc.) precisely so
run_campaign() logic stays decoupled from this subprocess machinery.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from avredteam_carla.agents.campaign import Trial, TRIAL_OUTCOME_INFRA_FAILURE
from avredteam_carla.evaluator import EpisodeMetrics
from avredteam_carla.runner import ScenarioConfig

# Matches verify_phase3.py's STAGE_SUBPROCESS_RETRIES - same node, same
# proven cap, no reason to pick a different number without new evidence.
DEFAULT_RETRIES = 3
# Matches verify_phase3.py's SETTLE_SLEEP_S (the settle delay Phase 3
# actually used, though not proven sufficient by itself - see
# docs/evaluator.md #8) as the starting backoff between attempts.
DEFAULT_BASE_BACKOFF_S = 8.0
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_MAX_BACKOFF_S = 60.0
# CARLA's own client-side load_world() timeout is 60s (docs/evaluator.md's
# root-cause investigation); a subprocess running a full episode needs far
# longer than that (Phase 3's real runs: ~600-700s), so this is a generous
# outer bound against a truly hung subprocess, not a per-RPC-call timeout.
DEFAULT_SUBPROCESS_TIMEOUT_S = 1800.0


def _unlink_if_exists(path: Path) -> None:
    # Path.unlink(missing_ok=True) is Python 3.8+ only; this project's
    # real env is 3.7 (same constraint as verify_phase3.py).
    if path.exists():
        path.unlink()


def _trial_from_dict(d: dict) -> Trial:
    metrics = EpisodeMetrics(**d["metrics"]) if d["metrics"] is not None else None
    return Trial(
        scenario_name=d["scenario_name"],
        attack_name=d["attack_name"],
        attack_params=d["attack_params"],
        metrics=metrics,
        outcome=d["outcome"],
        baseline_severity=d["baseline_severity"],
        error=d["error"],
    )


def run_trial_isolated(
    scenario: ScenarioConfig,
    attack_name: str,
    attack_params: dict,
    baseline_severity: float,
    *,
    retries: int = DEFAULT_RETRIES,
    base_backoff_s: float = DEFAULT_BASE_BACKOFF_S,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    max_backoff_s: float = DEFAULT_MAX_BACKOFF_S,
    subprocess_timeout_s: Optional[float] = DEFAULT_SUBPROCESS_TIMEOUT_S,
    sanity_frames_dir: Optional[str] = None,
    sleep: Callable[[float], None] = time.sleep,
    run_subprocess: Callable[..., "subprocess.CompletedProcess"] = subprocess.run,
) -> Trial:
    """Runs one trial in its own subprocess, retrying up to `retries`
    times with exponential backoff between attempts. Never raises for an
    ordinary trial failure - after exhausting retries, returns a
    Trial(outcome=infra_failure) instead (docs/search_methods.md
    "Failed-trial policy": a trial that exhausts retries is a distinct
    outcome, never recorded as severity_score=0, and this failed attempt
    still consumed one slot of the caller's budget - see that same
    section for why replacement was rejected).

    `sleep`/`run_subprocess` are injectable so this is unit-tested without
    real subprocess spawns or real multi-second waits.
    """
    scenario_json = json.dumps(dataclasses.asdict(scenario))
    attack_params_json = json.dumps(attack_params)

    last_error = None
    backoff = base_backoff_s

    for attempt in range(1, retries + 1):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = Path(f.name)

        cmd = [
            sys.executable, "-m", "avredteam_carla.agents.trial_worker",
            "--scenario-json", scenario_json,
            "--attack-name", attack_name,
            "--attack-params-json", attack_params_json,
            "--baseline-severity", str(baseline_severity),
            "--out", str(out_path),
        ]
        if sanity_frames_dir:
            cmd += ["--sanity-frames-dir", sanity_frames_dir]

        try:
            proc = run_subprocess(cmd, timeout=subprocess_timeout_s)
            if proc.returncode == 0 and out_path.exists():
                trial = _trial_from_dict(json.loads(out_path.read_text()))
                _unlink_if_exists(out_path)
                return trial
            last_error = f"subprocess exited {proc.returncode}"
        except subprocess.TimeoutExpired:
            last_error = f"subprocess timed out after {subprocess_timeout_s}s"
        except Exception as exc:  # noqa: BLE001 - a launch/parse failure is an infra failure, not a caller bug
            last_error = f"{type(exc).__name__}: {exc}"

        _unlink_if_exists(out_path)

        if attempt < retries:
            sleep(min(backoff, max_backoff_s))
            backoff *= backoff_factor

    return Trial(
        scenario_name=scenario.name,
        attack_name=attack_name,
        attack_params=dict(attack_params),
        metrics=None,
        outcome=TRIAL_OUTCOME_INFRA_FAILURE,
        baseline_severity=baseline_severity,
        error=last_error,
    )
