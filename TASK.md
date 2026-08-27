# Phase 1 task brief (for context when continuing this work on Maui)

This file preserves the original Phase 1 task brief and the decisions made
before any code here was executed on real hardware, so a session running on
the actual cluster doesn't have to re-derive them.

## Research context

Standalone research project studying whether an LLM agent can discover
sensor-level adversarial attacks on a learning-based autonomous driving
planner more efficiently than classical black-box search (random search,
Bayesian optimization). **Phase 1 only sets up simulation + planner
infrastructure — no attacks, no search methods, no LLM agent integration.**

Environment: lab compute cluster ("Maui"), multiple GPUs, some shared with
other users' jobs. GPU 0 may be under contention — check `nvidia-smi` and
prefer an idle GPU; GPU choice is configurable (env var / CLI flag), never
hardcoded.

## Goal for this phase

Get CARLA running headless on the cluster, get the Roach RL planner running
inference against it, and run one full clean (unperturbed) episode
end-to-end with logged control outputs and event sensors (collision, lane
invasion). No attacks, no agent, no search — just prove the pipeline works.

## Decisions made before cluster execution (see docs/setup.md §0 for why)

1. **CARLA 0.9.11**, not 0.9.15 as originally specified — Roach's checkpoint
   was only validated against 0.9.10.1/0.9.11.
2. **No camera/lidar sensor rig** — Roach's actual observation is a
   rasterized birdview (BEV) pseudo-sensor + scalar state, not a physical
   camera/lidar. Camera rigs in the Roach repo belong to CILRS, a different
   (imitation-learning) agent.
3. **Leaderboard-1.0-format route**, not 2.0 — Roach's `LeaderboardEnv` only
   understands the older route XML format; "Leaderboard 2.0" postdates this
   repo and isn't compatible with CARLA 0.9.11 anyway.
4. **Route chosen**: Town01, weather_group=`simple` (ClearNoon), route/task
   index 0 — smallest map, satisfies the brief's "pick any simple route."
5. **Checkpoint**: fetched via the `wandb` API from
   `iccv21-roach/trained-models/1929isj0` (the "Roach" run itself, per that
   repo's README table), not a plain file download.
6. **Directory convention**: all project/data/cache/output work lives under
   `/data/$USER/carla-redteam` (this project's user: `savyo`), never under
   `/home` (small/quota-limited on Maui). See docs/setup.md §1.

## Steps (original brief, unchanged)

1. Environment check: `nvidia-smi`, confirm Vulkan-capable driver, dedicated
   conda/micromamba env named `carla-redteam` (not a reused lab env).
2. Install CARLA (server + Python API) project-local, not system-wide.
   Document exact steps in `docs/setup.md`, reproducibly.
3. Launch CARLA headless (`-RenderOffScreen`), specific RPC port, pinned to a
   specific idle GPU via `CUDA_VISIBLE_DEVICES`. `scripts/launch_carla.sh`
   takes GPU id + port as arguments (parameterized for Phase 6's parallel
   instances).
4. Verify client connection: minimal script connects via `carla.Client`,
   calls `get_world()`, lists maps/blueprints.
5. Set up Roach: clone the repo, get its checkpoint, run its own eval/
   rollout tooling standalone against the CARLA server from step 4 — no
   custom integration code yet.
6. Custom minimal episode runner
   (`avredteam_carla/run_clean_episode.py`): spawns ego with Roach's actual
   expected observation config, runs one full route with zero perturbation,
   logs per-tick control + collision/lane-invasion events to CSV/JSON under
   `logs/`, cleans up actors/world state on exit.
7. Acceptance criteria (verify explicitly, report results — see the table at
   the end of `docs/setup.md`):
   - Script runs start-to-finish, no crashes
   - Route completes or terminates via a real collision (not an error)
   - Control outputs sane (steer/throttle/brake in range, not NaN/constant)
   - Log file inspectable, matches what you'd see in CARLA's spectator view

## Explicitly out of scope for this phase

No attacks, no search/optimization methods, no LLM agent integration.

## Git workflow

- Branch: `phase-1-carla-roach-setup`
- Commit incrementally as each step completes.
- Do not commit CARLA binaries, Roach checkpoints, or generated logs
  (`.gitignore` covers these) — code, configs, docs only.
- PR into `main`, titled "Phase 1: CARLA + Roach infrastructure", pasting
  the acceptance-criteria results. Do not merge without review.
- If anything requires a design decision not covered above or in
  docs/setup.md §0, stop and ask rather than guessing.
