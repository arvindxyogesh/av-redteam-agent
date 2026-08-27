"""Run one full clean (zero-perturbation) episode of Roach's RL expert
against a running CARLA server, logging per-tick control output plus
collision and lane-invasion events.

Phase 1 of this project is infrastructure only: no attacks, no search, no
agent. This script exists purely to prove the CARLA + Roach pipeline works
end to end.

This is a THIN wrapper around Roach's own tooling (per the phase-1 plan:
confirm the checkpoint + sensor config work with Roach's existing tooling
before writing custom integration code). Reused as-is from the `carla-roach`
repo (https://github.com/zhejz/carla-roach):
  - the registered `LeaderBoard-v0` gym env (`carla_gym`), which owns ego
    spawn, world ticking, the birdview observation manager, and the
    collision / outside-route-lane criteria modules
  - `agents.rl_birdview.rl_birdview_agent.RlBirdviewAgent`, which downloads
    the checkpoint from W&B and turns a birdview observation into a control
  - Roach's own reward/terminal entry points used for RL/benchmark runs:
    `reward.valeo_action:ValeoAction`, `terminal.valeo_no_det_px:ValeoNoDetPx`

What's new here: the tick loop, a hard guarantee that `control` is applied
unmodified (zero perturbation), CSV/JSON logging, and actor/world cleanup.

Event/termination-reason extraction was verified against a real run on Maui
(see docs/setup.md Sec 8) and against carla_gym source
(`ego_vehicle_handler.py:tick`, `criteria/collision.py`,
`criteria/outside_route_lane.py`, `terminal/valeo_no_det_px.py`):
`info_dict[actor_id]['collision']` / `['outside_route_lane']` are per-tick
top-level fields (falsy most ticks, a dict on the tick an event fires) -- NOT
nested under `'episode_event'` as originally guessed pre-hardware; that key
only appears on the final (done=True) tick and holds the whole episode's
accumulated event buffers. ValeoNoDetPx's own `terminal_debug` carries no
`'reason'` field at all (it's built for RL training debug text), so
`_termination_reason()` reconstructs the reason from those same buffers
instead of trusting a field that was never there.

See docs/setup.md for full installation + the reasoning behind the CARLA
0.9.11 (not 0.9.15) and birdview-not-camera decisions this script assumes.

Usage:
    python -m avredteam_carla.run_clean_episode \\
        --roach-root /data/savyo/carla-redteam/roach \\
        --host localhost --port 2000 \\
        --wb-run-path iccv21-roach/trained-models/1929isj0 \\
        --carla-map Town01 --weather-group simple --route-id 0 \\
        --out logs/clean_episode.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_clean_episode")

ACTOR_ID = "hero"
# Roach's control action space (see CarlaMultiAgentEnv.action_space):
# throttle in [0, 1], steer in [-1, 1], brake in [0, 1].
CONTROL_RANGES = {
    "throttle": (0.0, 1.0),
    "steer": (-1.0, 1.0),
    "brake": (0.0, 1.0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roach-root", required=True, help="Path to the cloned carla-roach repo")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument(
        "--wb-run-path",
        default="iccv21-roach/trained-models/1929isj0",
        help="W&B run path for the checkpoint to load (default: 'Roach' itself)",
    )
    p.add_argument("--wb-ckpt-step", default=None, help="Specific checkpoint step; default (None) = latest")
    p.add_argument("--carla-map", default="Town01", help="Must be one of Town01..Town06 (Roach's LeaderBoard maps)")
    p.add_argument(
        "--weather-group",
        default="simple",
        help="'simple' = single fixed weather (ClearNoon); see LeaderboardEnv.build_all_tasks for other groups",
    )
    p.add_argument(
        "--route-id",
        type=int,
        default=0,
        help="Index into the task list for the chosen map/weather-group. With --weather-group simple "
        "(a single weather) this is equivalent to the route index in that town's routes.xml.",
    )
    p.add_argument("--max-steps", type=int, default=6000, help="Safety cap (~10 min at 10Hz) in case a route never terminates")
    p.add_argument("--out", required=True, help="Output log path (.json or .csv)")
    p.add_argument("--workdir", default=None, help="Scratch dir for wandb/checkpoint downloads (default: a temp dir)")
    p.add_argument("--debug-dump-info", action="store_true", help="Print the raw per-tick info dict once, for schema verification")
    return p.parse_args()


def _write_seed_agent_config(path: Path, wb_run_path: str, wb_ckpt_step) -> None:
    """RlBirdviewAgent.setup() expects a local yaml with at least wb_run_path
    / wb_ckpt_step; it then overwrites its own config from the checkpoint's
    recorded config_agent.yaml downloaded from W&B. See rl_birdview_agent.py.
    """
    from omegaconf import OmegaConf

    OmegaConf.save(
        config=OmegaConf.create({"wb_run_path": wb_run_path, "wb_ckpt_step": wb_ckpt_step}),
        f=str(path),
    )


def _extract_events(info_for_actor: dict) -> dict:
    """Pull this tick's collision / lane-invasion events out of Roach's info
    dict. 'collision' and 'outside_route_lane' are top-level per-tick keys
    (see module docstring) - falsy on ticks with no event, a dict on the
    tick one actually fires.
    """
    collision = info_for_actor.get("collision")
    outside_route_lane = info_for_actor.get("outside_route_lane")
    return {
        "collision": [collision] if collision else [],
        "outside_route_lane": [outside_route_lane] if outside_route_lane else [],
    }


def _termination_reason(episode_event: dict) -> str:
    """Classify why an episode ended from Roach's accumulated per-episode
    event buffers (`info_dict[actor_id]['episode_event']`, only present on
    the final tick - see ego_vehicle_handler.py). Checked in roughly the
    same precedence ValeoNoDetPx.get() itself uses: blocked, red light,
    collision, stop sign, timeout, route completed. One of ValeoNoDetPx's
    own done conditions (lateral distance from the route too large) is
    computed inline in that class and never surfaces in any criterion
    buffer, so it can't be distinguished here - it falls through to
    'unknown' like any other unclassified case.
    """
    if episode_event.get("vehicle_blocked"):
        return "vehicle_blocked"
    if episode_event.get("red_light"):
        return "run_red_light"
    n_collisions = sum(
        len(episode_event.get(k, []))
        for k in ("collisions_layout", "collisions_vehicle", "collisions_pedestrian", "collisions_others")
    )
    if n_collisions:
        return "collision"
    if episode_event.get("stop_infraction"):
        return "run_stop_sign"
    if episode_event.get("timeout"):
        return "timeout"
    if episode_event.get("route_completion", {}).get("is_route_completed"):
        return "route_completed"
    return "unknown (see 'episode_event' in the output file for the raw buffers)"


def _check_control_sane(tick: int, control) -> list[str]:
    warnings = []
    for field, (lo, hi) in CONTROL_RANGES.items():
        val = getattr(control, field)
        if val != val:  # NaN check without importing math
            warnings.append(f"tick {tick}: {field} is NaN")
        elif not (lo <= val <= hi):
            warnings.append(f"tick {tick}: {field}={val} outside expected range [{lo}, {hi}]")
    return warnings


def main() -> int:
    args = parse_args()

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="roach_run_"))
    workdir.mkdir(parents=True, exist_ok=True)
    log.info("Using workdir (checkpoint/config downloads land here): %s", workdir)

    # Roach's modules (carla_gym, agents.*, reward.*, terminal.*) are not an
    # installed package - import them by adding the checkout to sys.path.
    sys.path.insert(0, args.roach_root)

    import gym
    import carla_gym  # noqa: F401  (side effect: registers LeaderBoard-v0 etc.)
    from agents.rl_birdview.rl_birdview_agent import RlBirdviewAgent

    seed_cfg_path = workdir / "config_agent.yaml"
    _write_seed_agent_config(seed_cfg_path, args.wb_run_path, args.wb_ckpt_step)

    prev_cwd = os.getcwd()
    os.chdir(workdir)  # wandb + RlBirdviewAgent download files relative to cwd
    try:
        log.info("Loading Roach agent from %s (this downloads the checkpoint via wandb on first run)", args.wb_run_path)
        agent = RlBirdviewAgent(path_to_conf_file=str(seed_cfg_path.name))
    finally:
        os.chdir(prev_cwd)

    agent_log_path = workdir / "agent.log"
    agent.reset(str(agent_log_path))

    obs_configs = {ACTOR_ID: agent.obs_configs}
    reward_configs = {ACTOR_ID: {"entry_point": "reward.valeo_action:ValeoAction", "kwargs": {}}}
    terminal_configs = {ACTOR_ID: {"entry_point": "terminal.valeo_no_det_px:ValeoNoDetPx", "kwargs": {}}}

    log.info(
        "Creating LeaderBoard-v0 env: map=%s weather_group=%s host=%s port=%d",
        args.carla_map, args.weather_group, args.host, args.port,
    )
    env = gym.make(
        "LeaderBoard-v0",
        obs_configs=obs_configs,
        reward_configs=reward_configs,
        terminal_configs=terminal_configs,
        carla_map=args.carla_map,
        host=args.host,
        port=args.port,
        seed=args.seed,
        no_rendering=True,
        weather_group=args.weather_group,
        routes_group=None,
    )

    ticks: list[dict] = []
    warnings: list[str] = []
    termination_reason = "unknown"

    try:
        env.set_task_idx(args.route_id)  # pin an exact route deterministically, no shuffling
        log.info("Pinned task_idx=%d (task=%s)", args.route_id, env.task)

        obs_dict = env.reset()
        timestamp = env.timestamp
        done_dict = {"__all__": False}
        tick_idx = 0
        final_episode_event = None

        while not done_dict["__all__"] and tick_idx < args.max_steps:
            control = agent.run_step(obs_dict[ACTOR_ID], timestamp)
            # Zero perturbation: control is applied exactly as Roach produced it.
            control_dict = {ACTOR_ID: control}

            obs_dict, reward_dict, done_dict, info_dict = env.step(control_dict)
            timestamp = env.timestamp

            if args.debug_dump_info and tick_idx == 0:
                log.info("First tick raw info_dict[%s]: %s", ACTOR_ID, info_dict[ACTOR_ID])

            warnings.extend(_check_control_sane(tick_idx, control))

            events = _extract_events(info_dict[ACTOR_ID])
            ticks.append(
                {
                    "tick": tick_idx,
                    "sim_time": timestamp.get("relative_simulation_time"),
                    "steer": control.steer,
                    "throttle": control.throttle,
                    "brake": control.brake,
                    "reward": reward_dict.get(ACTOR_ID),
                    "done": bool(done_dict.get(ACTOR_ID)),
                    "collision_events": events["collision"],
                    "outside_route_lane_events": events["outside_route_lane"],
                }
            )

            if done_dict.get(ACTOR_ID):
                final_episode_event = info_dict[ACTOR_ID].get("episode_event", {})
                termination_reason = _termination_reason(final_episode_event)

            tick_idx += 1

        if tick_idx >= args.max_steps and not done_dict["__all__"]:
            termination_reason = "max_steps_reached"

        final_stat = info_dict[ACTOR_ID].get("episode_stat") if ticks else None

    finally:
        # Always clean up actors/world state, even on failure.
        log.info("Closing env (destroys ego + zombie actors, resets world settings)")
        env.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "meta": {
            "carla_map": args.carla_map,
            "weather_group": args.weather_group,
            "route_id": args.route_id,
            "wb_run_path": args.wb_run_path,
            "n_ticks": len(ticks),
            "termination_reason": termination_reason,
            "sanity_warnings": warnings,
            "final_episode_stat": final_stat,
            "final_episode_event": final_episode_event,
            "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "ticks": ticks,
    }

    if out_path.suffix == ".csv":
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(ticks[0].keys()) if ticks else [])
            writer.writeheader()
            for row in ticks:
                writer.writerow(row)
        meta_path = out_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(result["meta"], indent=2))
        log.info("Wrote %d ticks to %s, meta to %s", len(ticks), out_path, meta_path)
    else:
        out_path.write_text(json.dumps(result, indent=2, default=str))
        log.info("Wrote %d ticks to %s", len(ticks), out_path)

    log.info("Termination reason: %s", termination_reason)
    if warnings:
        log.warning("%d control sanity warnings (see log file 'sanity_warnings')", len(warnings))
    else:
        log.info("No control sanity warnings - all steer/throttle/brake values in expected range.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
