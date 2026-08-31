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

---

# Phase 2 task brief — BEV-space sensor attack library

Following Phase 1 (merged: CARLA 0.9.11 + Roach headless on Maui, clean
episodes verified). Correction from Phase 1: Roach's `RlBirdviewAgent`
consumes a rasterized birdview (BEV) image + scalar state — not camera/
lidar. "Sensor-level attack" means perturbing that BEV raster and scalar
state, injected between `carla_gym`'s observation manager and the policy's
forward pass — not attacking raw CARLA actors or camera buffers.

## Goal

A small library of attacks on Roach's actual observation space (BEV raster
+ scalar state), each verified to measurably change Roach's control output
vs. a clean baseline, with visual (PNG) sanity checks since the cluster run
is headless.

## What was built (see `docs/attacks.md` for the full writeup)

1. `docs/attacks.md` §1-4 — the real BEV/scalar-state layout and the exact
   interception point, read from `carla-roach` source (chauffeurnet.py,
   rl_birdview_wrapper.py, rl_birdview_agent.py), not the paper.
2. `avredteam_carla/attacks/base.py` — `Attack` base class +
   `TunableParam` declarative schema (name/type/range/default), the shared
   contract later phases' search methods will use.
3. Three attacks: `ChannelNoiseAttack`, `GeometrySpoofAttack`,
   `PhantomActorAttack` (`avredteam_carla/attacks/{channel_noise,
   geometry_spoof,phantom_actor}.py`), each unit-tested against synthetic
   BEV tensors (`tests/test_attacks.py`) without needing CARLA.
4. `avredteam_carla/attacks/hook.py` — monkeypatches
   `RlBirdviewWrapper.process_obs` (the exact interception point) rather
   than forking Roach's code.
5. `avredteam_carla/run_clean_episode.py` — extended with `--attack` /
   `--attack-param` / `--bev-frames-every`; the no-attack path is
   unchanged from Phase 1.
6. `avredteam_carla/compare_episodes.py` — deviation metrics (steering
   sign-flips, speed/brake stats, paired control divergence) between two
   episode logs.

## Explicitly out of scope for this phase

No search/optimization methods (random search, Bayesian opt, LLM agent) —
that's Phase 4. No formal evaluator/severity scoring — that's Phase 3.

## Git workflow

- Branch: `phase-2-bev-attack-library`
- PR titled "Phase 2: BEV-space sensor attack library", acceptance table
  filled in with real Maui results. Do not merge without review.
- Don't bulk-commit BEV frame PNGs — `.gitignore` already excludes
  `logs/` (where `--bev-frames-every` writes them) except `.gitkeep`.

---

# Phase 3 task brief — Evaluator + Trial/CampaignResult runner

Following Phase 1 (merged) and Phase 2 (PR #2, verified). Turns "attack
visibly changes control output" into formal metrics, and wraps everything
into the `Trial`/`CampaignResult` interface Phase 4's search methods
(random search, Bayesian opt, LLM agent) will all share.

## What was built (see `docs/evaluator.md` for the full writeup)

1. `docs/evaluator.md` — formal metric definitions (chattering rate,
   steering jerk, route/lane deviation, time-to-collision-or-completion,
   braking severity, composite `severity_score`), each with exact formula,
   units, and an explicit Nyquist/aliasing argument for the rate-based
   metrics (the Phase 2 aliasing bug's lesson, applied here).
2. A real "stop and ask" resolved: `carla_gym`'s ground-truth lateral-
   distance signal (`ValeoNoDetPx`'s own `lat_dist`) exists but only ever
   gets baked into a debug string, never a clean field. Confirmed with the
   project owner to replicate the same formula from the same ground-truth
   inputs as a new logged field (`lateral_offset_m`), rather than parsing
   debug text or deriving from the attacked BEV raster.
3. `avredteam_carla/evaluator.py` — `EpisodeMetrics` + `evaluate(log)`,
   pure Python, 15 unit tests against synthetic logs.
4. `avredteam_carla/agents/campaign.py` — `Trial`/`CampaignResult` +
   `sorted_by_severity()`, exactly the shape specified, 6 unit tests.
5. `avredteam_carla/ground_truth.py` — real-CARLA-state computations for
   route deviation and obstacle clearance, neither read from any existing
   `info_dict` field.
6. `run_clean_episode.py` refactored: `run_episode()` is now importable
   (the CLI's `main()` is a thin wrapper around it), used by
   `avredteam_carla/runner.py`'s `run_trial()`/`run_baseline()`.
7. `avredteam_carla/verify_phase3.py` — runs baseline + all three Phase 2
   attacks + a repeated-`run_trial`-call stability check, prints the
   acceptance table.

## Explicitly out of scope for this phase

No search/optimization methods (random search, Bayesian opt, LLM agent) —
Phase 4. No scenario suite beyond the single Town01 route — Phase 5.

## Git workflow

- Branch: `phase-3-evaluator-campaign-runner`
- PR titled "Phase 3: Evaluator + campaign runner", acceptance table filled
  in with real Maui results, stability-check and aliasing-check results
  explicitly called out. Do not merge without review.

---

# Phase 4 task brief — Search methods (random search, Bayesian opt, LLM agent)

Following Phase 1-3 (PRs #1-3, unmerged). Adds three search methods that
all call `run_trial()` in a loop, sharing one interface
(`SearchMethod.run_campaign()`), so they're directly comparable.

## What was built (see `docs/search_methods.md` for the full writeup)

1. **Step 0, length-normalization — verified, not assumed.** Checked
   `evaluator.py`'s real code: `max_lateral_offset`/`min_obstacle_clearance`
   are running max/min over the whole episode, so their length-sensitivity
   is a mathematical certainty (a min over a longer prefix can only stay
   equal or decrease), unlike the already rate-normalized `off_lane_frac`/
   `chattering_rate`. Locked in with unit tests on synthetic data
   (`tests/test_analyze_episode_length_bias.py`); the real *magnitude* on
   Phase 3's actual logs needs a real-hardware follow-up (no log files
   exist in this dev sandbox - gitignored, live only on Maui). No formula
   change applied yet - `docs/search_methods.md` §0 has the two candidate
   fixes to test once real magnitude data exists.
2. **`max_brake_rate` "fix" — checked, turned out to be a no-op.** The
   brief's instruction to drop it from `severity_score`'s composite
   assumed it was a term there; checked the real formula first and found
   it never was (`docs/evaluator.md` #7 already excluded it). No formula
   change made; documented the check in `docs/evaluator.md` and
   `docs/search_methods.md` rather than silently "fixing" something that
   wasn't broken.
3. `Trial` gains `delta_severity` (the real search objective -
   `severity_score - baseline_severity`) and `outcome` (`success`/
   `infra_failure`, so a trial that exhausts its retries is never recorded
   as `severity_score=0`) - `avredteam_carla/agents/campaign.py`.
4. `avredteam_carla/agents/search.py` — the shared `SearchMethod`
   interface + `attack_pool()`/`sample_uniform_params()`/
   `tunable_params_for()` helpers, reading `ATTACK_REGISTRY`
   programmatically.
5. Three search methods, each unit-tested against a stubbed `TrialRunner`
   (no CARLA needed): `avredteam_carla/agents/random_search.py`,
   `bayesian_search.py` (Optuna, one TPE study per campaign, attack type
   as a categorical param - design choice documented in
   `docs/search_methods.md` §6), `llm_agent_search.py` (Claude tool-use
   loop, budget enforced both directions, falls back to random sampling
   if the model stalls/misbehaves - tested against a scripted fake
   Anthropic client).
6. Infra hardening, applied to all three methods via one production
   `TrialRunner`: `avredteam_carla/agents/isolated_runner.py`
   (subprocess-per-trial via `agents/trial_worker.py`, retry-with-backoff
   - 3 attempts, 8s/16s/... capped at 60s, matching Phase 3's own proven
   numbers), `avredteam_carla/preflight.py` (disk/GPU/load snapshot once
   per campaign + automatic idle-GPU pick), `avredteam_carla/attacks/
   sanity_frames.py` (3 representative BEV frames per trial via an online
   doubling/proxy-score heuristic, not a full-episode dump). Failed-trial
   policy decided: an `infra_failure` trial consumes budget rather than
   being replaced (bounded worst-case runtime on a node Phase 3 already
   found bursty).
7. `avredteam_carla/verify_phase4.py` — runs all three methods on Town01
   with a small shared budget + fixed seed, checks every output path
   actually resolves under `/data/$USER` before running. Needs a live
   CARLA server + `optuna`/`anthropic` installed + `ANTHROPIC_API_KEY` -
   not runnable in this dev sandbox; the acceptance table needs a
   real-hardware follow-up, same as every prior phase's real numbers.

## Explicitly out of scope for this phase

No scenario suite beyond Town01/route-0 (Phase 5). No full budget x seed x
scenario sweep at scale (Phase 6) - this phase proves the three methods
work end-to-end, Phase 6 is where they run at the scale the paper needs.

## Git workflow

- Branch: `phase-4-search-methods`, cut from the tip of
  `phase-3-evaluator-campaign-runner`.
- PR titled "Phase 4: Search methods (random, Bayesian, LLM agent)",
  acceptance table filled in with real results from a real-hardware
  follow-up session (Step 0's real length-bias magnitude and Step 6's
  verification table both need one - see `docs/search_methods.md`). Do
  not merge without review.
