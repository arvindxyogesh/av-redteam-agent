"""Worker entrypoint for one isolated trial - re-invoked as a fresh
subprocess by avredteam_carla.agents.isolated_runner.run_trial_isolated()
(docs/search_methods.md Step 5's "subprocess-per-trial isolation -
mandatory, not optional").

Runs exactly one avredteam_carla.runner.run_trial() call and writes its
Trial as JSON to --out. Not meant to be invoked directly by a human -
mirrors the self-reinvocation shape verify_phase3.py already proved works
for exactly this reason (Phase 3's real finding: repeated in-process
run_trial() calls can abort the whole process via an uncaught C++
exception no Python try/except can catch - a fresh process per trial is
the only complete fix, see docs/evaluator.md #8's stability-check section).

Usage (normally only from isolated_runner.py, not by hand):
    python -m avredteam_carla.agents.trial_worker \\
        --scenario-json '{"name": "...", "roach_root": "...", ...}' \\
        --attack-name channel_noise \\
        --attack-params-json '{"amplitude": 50.0}' \\
        --baseline-severity 16.1 \\
        --out /data/savyo/carla-redteam/campaigns/.../trial_003.json \\
        [--sanity-frames-dir /data/savyo/carla-redteam/campaigns/.../trial_003/frames]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("trial_worker")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario-json", required=True, help="JSON-serialized ScenarioConfig (dataclasses.asdict)")
    p.add_argument("--attack-name", required=True)
    p.add_argument("--attack-params-json", default="{}")
    p.add_argument("--baseline-severity", type=float, required=True)
    p.add_argument("--sanity-frames-dir", default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from avredteam_carla.runner import ScenarioConfig, run_trial

    scenario = ScenarioConfig(**json.loads(args.scenario_json))
    attack_params = json.loads(args.attack_params_json)

    trial = run_trial(
        scenario,
        args.attack_name,
        attack_params,
        baseline_severity=args.baseline_severity,
        sanity_frames_dir=args.sanity_frames_dir,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trial.to_dict(), indent=2, default=str))
    log.info("Wrote trial result (outcome=%s) to %s", trial.outcome, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
