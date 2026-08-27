# Phase 1 setup: CARLA + Roach on Maui

This document is the reproducible, step-by-step record of how the CARLA + Roach
pipeline was (or should be) installed on the Maui cluster. Follow it top to
bottom on a fresh account; each step notes what to verify before moving on.

## 0. Key decisions (read this first)

- **CARLA version: 0.9.11, not 0.9.15.** The task brief for this phase
  originally called for CARLA 0.9.15. Roach's own checkpoint and evaluation
  tooling (`carla-roach` repo) were only ever validated against CARLA 0.9.10.1
  / 0.9.11 — the repo's `doc/INSTALL.md` explicitly pins 0.9.11 for general use
  and 0.9.10.1 for the RL expert specifically ("0.9.11 crashes more often for
  unknown reasons" during RL). CARLA 0.9.15 postdates this repo and was never
  tested against it. We deviate from the literal brief and use **0.9.11** to
  maximize the odds of a clean, correctly-behaving Phase 1 baseline. This is
  the single most important thing to re-verify if inference looks wrong.
- **Sensor rig: no camera, no lidar.** The task brief assumed Roach uses a
  camera+lidar sensor rig. It does not. The actual "Roach" agent
  (`RlBirdviewAgent`, entry point `agents.rl_birdview.rl_birdview_agent`) is a
  PPO policy trained on a **rasterized birdview (BEV) pseudo-sensor**
  (`carla_gym/core/obs_manager/birdview`, 192×192 px, 5 px/m,
  `pixels_ev_to_bottom=40`, 4 history frames — see
  `config/agent/ppo/obs_configs/birdview.yaml` in the Roach repo) plus scalar
  speed/control/velocity state. Camera rigs in that repo
  (`central_rgb`, `three_rgb_wide`, etc.) belong to **CILRS**, a separate
  imitation-learning baseline that is *not* Roach. Do not attach a camera or
  lidar sensor for Roach inference — it isn't used and won't affect the
  policy's actions.
- **Route: CARLA Leaderboard 1.0 route format, not 2.0.** "Leaderboard 2.0"
  (OpenSCENARIO routes, Town12/13, newer CARLA) postdates this repo and is not
  compatible with CARLA 0.9.11 or with Roach's `LeaderboardEnv`. We use
  Roach's own `LeaderBoard-v0` gym env, which loads Leaderboard-1.0-style
  route XML from `carla_gym/envs/scenario_descriptions/LeaderBoard/<Town>/`.
  Picked route: **Town01, weather_group=`simple` (ClearNoon), route_id 0** —
  Town01 is the smallest map and matches the task's "pick any simple route"
  allowance.
- **Checkpoint delivery: W&B, not a file download.** Roach's checkpoint is
  fetched at runtime via the `wandb` API from the public project
  `iccv21-roach/trained-models`, run path `1929isj0` (this is literally
  "Roach" per that repo's README table). This needs `wandb login` once.

If inference looks wrong or crashes in a way that traces back to any of these
choices, stop and re-open the discussion rather than silently patching around
it — these were explicit trade-offs, not guesses.

## 1. Directory layout — everything under `/data/$USER`

`/home` on Maui is small/quota-limited; all project code, conda envs, package
caches, CARLA/Roach installs, and generated outputs must live under
`/data/$USER`. Only the git checkout of *this* repo needs to exist anywhere
convenient (it's small); everything else below assumes:

```bash
export PROJECT_DATA_DIR=/data/$USER/carla-redteam
mkdir -p "$PROJECT_DATA_DIR"
```

(For this project, `$USER` is `savyo`, so `PROJECT_DATA_DIR=/data/savyo/carla-redteam`.)

Layout once set up:

```
/data/$USER/carla-redteam/
├── envs/            # micromamba env prefix (or conda pkgs cache)
├── pkgs/            # package cache
├── CARLA_0.9.11/    # CARLA server + Python API
├── roach/           # carla-roach checkout + downloaded checkpoint
├── logs/            # episode logs (symlink target for repo's logs/ if desired)
└── cache/           # pip / wandb / HF cache etc.
```

Redirect every cache that defaults to `$HOME` before installing anything:

```bash
export XDG_CACHE_HOME="$PROJECT_DATA_DIR/cache"
export PIP_CACHE_DIR="$PROJECT_DATA_DIR/cache/pip"
export CONDA_PKGS_DIRS="$PROJECT_DATA_DIR/pkgs"
export WANDB_DIR="$PROJECT_DATA_DIR/cache/wandb"
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$WANDB_DIR"
```

Add these `export`s (with `$PROJECT_DATA_DIR` set) to your shell profile or a
project-specific env file you source before working, so they're never
forgotten.

## 2. GPU check

```bash
nvidia-smi
```

Record driver version and confirm it's Vulkan-capable (any driver that
supports CUDA 11+ on a Turing-or-newer GPU is fine for CARLA 0.9.11's UE4
renderer). Check `nvidia-smi` for per-GPU utilization/memory before picking a
GPU — prefer one that's idle. GPU selection is passed explicitly to every
script below via `CUDA_VISIBLE_DEVICES` / a `--gpu` flag; nothing is
hardcoded to GPU 0.

**TODO (fill in after running on Maui):** exact GPU model(s), driver version,
and which GPU id was idle/selected.

## 3. Create the conda/micromamba environment

Use micromamba (no admin rights needed, faster than conda) installed under
`$PROJECT_DATA_DIR`:

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$PROJECT_DATA_DIR" bin/micromamba
export MAMBA_ROOT_PREFIX="$PROJECT_DATA_DIR/envs"
alias micromamba="$PROJECT_DATA_DIR/bin/micromamba"
```

Roach pins exact dependency versions (PyTorch, gym, stable-baselines3, hydra,
omegaconf, wandb) in its own `environment.yml` — reuse that file rather than
re-deriving versions, but override the environment name to keep this project
self-contained:

```bash
git clone https://github.com/zhejz/carla-roach.git "$PROJECT_DATA_DIR/roach"
cd "$PROJECT_DATA_DIR/roach"
micromamba env create -n carla-redteam -f environment.yml
micromamba activate carla-redteam
```

Roach's `environment.yml` pins **Python 3.7**. If micromamba can't fully
solve the AWS-era pinned env as-is, fall back to `conda env create` with the
same file/name, or relax to `python=3.7` + a fresh `pip install` of the
unpinned subset actually needed (`torch`, `gym==0.21.*`, `hydra-core`,
`omegaconf`, `wandb`, `stable-baselines3`, `carla` client wheel) — but try the
pinned file first since Roach's checkpoint loading is version-sensitive.

## 4. Install CARLA 0.9.11 (server + Python API)

Project-local, under `$PROJECT_DATA_DIR` (not system-wide):

```bash
mkdir -p "$PROJECT_DATA_DIR/CARLA_0.9.11" && cd "$PROJECT_DATA_DIR/CARLA_0.9.11"
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz
tar -xvzf CARLA_0.9.11.tar.gz
mkdir -p Import
wget -P Import https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.11.tar.gz
bash ImportAssets.sh
rm CARLA_0.9.11.tar.gz Import/AdditionalMaps_0.9.11.tar.gz

export CARLA_ROOT="$PROJECT_DATA_DIR/CARLA_0.9.11"
```

Verify the S3 URLs are still live before relying on them — CARLA has moved
release hosting before. If `carla-releases.s3.eu-west-3.amazonaws.com` 404s,
check https://github.com/carla-simulator/carla/releases/tag/0.9.11 for the
current asset links.

Install the Python client API into the `carla-redteam` env (egg filename must
match the Python version the env actually has — 3.7 per Roach's pinned env):

```bash
micromamba activate carla-redteam
easy_install "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg"
# or, if that env ships pip-friendly wheels instead:
pip install "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.11-cp37-cp37m-linux_x86_64.whl"
```

**TODO (fill in after running on Maui):** which of the two installs above
actually worked (`easy_install` vs `pip install` — depends on what CARLA
0.9.11 ships in `PythonAPI/carla/dist/`).

## 5. Launch CARLA headless

Use `scripts/launch_carla.sh <gpu_id> <port> [carla_root]` (see that file —
parameterized so Phase 6 can launch several instances in parallel on
different GPUs/ports). Example:

```bash
CARLA_ROOT="$PROJECT_DATA_DIR/CARLA_0.9.11" ./scripts/launch_carla.sh 1 2000
```

This pins the server to GPU 1 via `CUDA_VISIBLE_DEVICES`, runs
`-RenderOffScreen` (headless), and listens for RPC on port 2000.

**TODO (fill in after running on Maui):** confirm the server actually starts
and stays up (check the log file the script writes) before moving on.

## 6. Verify client connection

```bash
python scripts/check_client_connection.py --host localhost --port 2000
```

This connects via `carla.Client`, calls `get_world()`, and lists available
maps and blueprints. **TODO:** paste the actual output here once run.

## 7. Run Roach's own eval tooling standalone (no custom code)

Confirm the checkpoint + sensor config work with Roach's own tooling before
writing anything custom, per the phase-1 plan. From `$PROJECT_DATA_DIR/roach`:

```bash
micromamba activate carla-redteam
export CARLA_ROOT="$PROJECT_DATA_DIR/CARLA_0.9.11"
wandb login   # one-time; needs a free wandb.ai account
```

Edit `run/benchmark.sh` to use the RL-expert ("Roach") block instead of the
default Autopilot block (the file has both, Autopilot uncommented by
default) — set:

```bash
agent="ppo"
# ...
agent.ppo.wb_run_path=iccv21-roach/trained-models/1929isj0
test_suites=lb_test_tt   # or nocrash_dense — either is fine for a smoke test
carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
```

Then run it:

```bash
bash run/benchmark.sh
```

**TODO (fill in after running on Maui):** did the checkpoint download via
wandb succeed, did the benchmark run start-to-finish, and do the logged
episode stats look sane (non-zero route completion, no immediate crash)?

## 8. Custom minimal clean-episode runner

Once step 7 confirms the checkpoint + env work as-is, run the project's own
minimal runner (`avredteam_carla/run_clean_episode.py`), which reuses Roach's
own `carla_gym` environment and `RlBirdviewAgent` rather than hand-rolling
sensor/actor code:

```bash
python -m avredteam_carla.run_clean_episode \
  --carla-root "$CARLA_ROOT" \
  --roach-root "$PROJECT_DATA_DIR/roach" \
  --host localhost --port 2000 \
  --wb-run-path iccv21-roach/trained-models/1929isj0 \
  --carla-map Town01 --weather-group simple --route-id 0 \
  --out logs/clean_episode_$(date +%Y%m%d_%H%M%S).json
```

See that script's docstring for exactly which of Roach's own building blocks
it reuses (`gym.make('LeaderBoard-v0', ...)`, `RlBirdviewAgent`, and Roach's
own `reward.valeo_action:ValeoAction` / `terminal.valeo_no_det_px:ValeoNoDetPx`
entry points) versus what's new (the tick loop, CSV/JSON logging, cleanup).

## 9. Acceptance criteria — fill in after a real run

| Criterion | Result |
|---|---|
| Script runs start-to-finish, no crash | TODO |
| Route completes or terminates via real collision (not error) | TODO |
| Control outputs sane (steer/throttle/brake in range, not NaN/constant) | TODO |
| Log file inspectable, matches spectator-view expectations | TODO |
| GPU actually used (nvidia-smi during run) and which one | TODO |

Do not mark Phase 1 done in the PR description until every row above is
filled in with a real, observed result — not an expectation.
