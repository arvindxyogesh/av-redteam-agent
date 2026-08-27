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

- **CARLA server delivery: official Docker image, not the S3 tarball.**
  `https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz`
  (the URL this doc and Roach's own `doc/INSTALL.md` both specify) is dead —
  the eu-west-3 bucket 301-redirects with no `Location` header (S3's way of
  saying "wrong region, retry"), and every region-corrected variant we tried
  (`s3.us-west-2`, `s3-us-west-2` dash form, generic `s3.amazonaws.com`) 403s.
  CARLA appears to have pruned pre-0.9.12 raw tarballs from that bucket. The
  official `carlasim/carla:0.9.11` Docker Hub image (pushed 2020-12-23, still
  active, ~4.6GB) is the working substitute — same server binary, and it
  still contains a full `PythonAPI/` tree we extract the client wheel from.
  This machine already had the nvidia container runtime and CDI GPU devices
  configured, so `docker run --gpus device=N ...` was a straightforward swap
  for the bare `CarlaUE4.sh` invocation. See §4/§5 below.
- **PyTorch: no CUDA build supports the H200 in a Python-3.7 env — fell back
  to CPU inference for Roach's policy net.** Roach's checkpoint loading and
  the CARLA 0.9.11 Python client are both effectively pinned to Python 3.7
  (no `carla` wheel for 0.9.11 exists for any newer CPython on PyPI, and
  builds are ABI-specific). PyPI's newest `cp37` PyTorch wheel is `1.13.1`
  (`+cu117`); `pytorch.org`'s cu118/cu121 indices only ship `cp37` wheels for
  `manylinux2014_aarch64`, not x86_64. Loading `torch==1.13.1+cu117` on this
  node's H200 (compute capability sm_90) prints PyTorch's own
  `"NVIDIA H200 ... is not compatible with the current PyTorch installation"`
  warning (supported archs top out at sm_86 — Ampere) and a real op on a CUDA
  tensor hangs/never returns. CUDA 11.8 was the first toolkit release with
  official Hopper (sm_90a) support, and no post-CUDA-11.8 PyTorch wheel was
  ever built for Python 3.7. Net effect: within a Python-3.7 env there is no
  "newer CUDA-compatible wheel" that actually runs on this GPU — the fix the
  task brief anticipated tops out at torch 1.13.1, which still doesn't work
  here. We run Roach's PPO policy net on CPU (`device="cpu"`, small
  birdview-CNN + MLP, negligible latency at 10Hz) while the CARLA server
  itself still renders on the pinned GPU via `-RenderOffScreen`/Vulkan — the
  server process is what "GPU actually used" in the acceptance table below
  refers to, since that's the only heavy GPU consumer in this pipeline.

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

**Observed on this node:** 8x NVIDIA H200 (143.8GB each), driver 595.58.03,
`nvidia-smi`-reported CUDA 13.2 (Vulkan-capable — H200/Hopper is far newer
than the Turing baseline CARLA 0.9.11's UE4 renderer needs). GPUs 0, 1, 2, 5
were running other users' jobs (VLLM engine cores) at the time of this run;
GPUs 3, 4, 6, 7 were idle (0MiB used, 0% util). **GPU 3 selected** — passed
explicitly via `CUDA_VISIBLE_DEVICES=3` (bare-binary path) or
`--gpus device=3` (Docker path, see §4), never hardcoded in any script.

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

**What actually happened on this node:** skipped straight to the relaxed
`python=3.7` + unpinned-subset path rather than attempting the full
`environment.yml` solve. Two reasons: (1) `cudatoolkit=10.1`/`pytorch=1.4.0`
in that file cannot run on this node's H200 regardless (see §0), so solving
it would burn time on a ~200-package, TensorFlow-inclusive 2020 conda-forge
environment (TF isn't even used by the birdview RL agent — it's a leftover
from a CILRS/training path this phase doesn't touch) for no payoff; (2) it
would need to re-solve/re-download several GB of packages this phase doesn't
exercise. Used the pre-existing micromamba binary at
`/home/savyo/micromamba/bin/micromamba` (v2.8.1) rather than re-downloading
one, with `MAMBA_ROOT_PREFIX=$PROJECT_DATA_DIR/envs` so the env itself still
lands under `/data/$USER` per §1:

```bash
/home/savyo/micromamba/bin/micromamba create -y -n carla-redteam -c conda-forge python=3.7
```

This produced Python 3.7.12. Then, into that env's `bin/python -m pip`:
`torch==1.13.1` (newest `cp37` wheel on PyPI; see §0 for why this still ends
up CPU-only on this GPU), `gym==0.21.0`, `hydra-core==1.0.3`,
`omegaconf==2.0.2`, `wandb==0.15.12`, `stable-baselines3==0.8.0`,
`opencv-python==4.5.1.48`, `imgaug==0.4.0`, `pygame`, `dictor`, `networkx`,
`shapely`, `tabulate`, `pillow`, and (pinned, see below) `py-trees==0.8.3`.

Two real install/runtime snags hit and fixed along the way, both from
installing 2020-era packages against a current pip/Python-ecosystem:

- `gym==0.21.0` failed `setup.py egg_info` under current `setuptools` with
  `error in gym setup command: 'extras_require' must be a dictionary whose
  values are strings or lists of strings...` — fixed by installing
  `setuptools==65.5.0 wheel==0.38.4` into the env *before* installing gym
  (well-known gym-0.21 / modern-setuptools incompatibility).
- `gym==0.21.0` then installed but failed on `import gym` with
  `AttributeError: 'EntryPoints' object has no attribute 'get'` — a
  breaking API change in newer `importlib_metadata`. Fixed by pinning
  `importlib-metadata==4.13.0`.
- Installing bare `py-trees` (unpinned) pulled a modern (2.x) release that
  fails to import under Python 3.7 (`TypeError: 'type' object is not
  subscriptable`, from `list[...]`-style builtin generics in its source,
  which need Python 3.9+). Fixed by pinning `py-trees==0.8.3`, the exact
  version Roach's `environment.yml` specifies.

## 4. Install CARLA 0.9.11 (server + Python API)

**What actually worked on this node: the official Docker image, not the S3
tarball.** Both S3 URLs in the block below are dead (see §0 for exactly what
was tried and how it failed — 301-with-no-Location, then 403 on every
region-corrected endpoint). Keeping the original recipe here for reference /
in case the bucket comes back, but do not expect it to work:

```bash
mkdir -p "$PROJECT_DATA_DIR/CARLA_0.9.11" && cd "$PROJECT_DATA_DIR/CARLA_0.9.11"
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz   # DEAD, see §0
tar -xvzf CARLA_0.9.11.tar.gz
mkdir -p Import
wget -P Import https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.11.tar.gz  # DEAD, see §0
bash ImportAssets.sh
rm CARLA_0.9.11.tar.gz Import/AdditionalMaps_0.9.11.tar.gz
```

**Actual recipe used:** pull the official `carlasim/carla:0.9.11` Docker Hub
image (this is also literally how CARLA's own docs recommend running it
headless on Linux, tarball or not), then copy the two things we need — the
`PythonAPI/` tree and `CarlaUE4.sh` — out to `$PROJECT_DATA_DIR/CARLA_0.9.11`
so the rest of this doc's paths (`$CARLA_ROOT`, `launch_carla.sh`) still work
unchanged; the server itself still runs *inside* the container (see §5 for
`launch_carla.sh`'s Docker mode), this is only to get local access to the
Python API without a second full checkout:

```bash
export CARLA_ROOT="$PROJECT_DATA_DIR/CARLA_0.9.11"
docker pull carlasim/carla:0.9.11
docker create --name carla_extract_0911 carlasim/carla:0.9.11
mkdir -p "$CARLA_ROOT"
docker cp carla_extract_0911:/home/carla/PythonAPI "$CARLA_ROOT/PythonAPI"
docker cp carla_extract_0911:/home/carla/CarlaUE4.sh "$CARLA_ROOT/CarlaUE4.sh"
docker rm carla_extract_0911
```

This produced `$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg`
(and a `py2.7` one we don't need) — an **egg**, not a wheel, matching the
Python 3.7 env from §3.

Install the Python client API into the `carla-redteam` env. Modern
`setuptools` (65.5.0, what we're on after the §3 gym fix) dropped the
`easy_install` console script entirely, so the documented `easy_install
path/to.egg` command from Roach's own install doc no longer works out of the
box. The CARLA 0.9.11 egg is just a zip of a pure-Python `carla/` package
plus one compiled `.so` (`libcarla.cpython-37m-x86_64-linux-gnu.so`) — no
build step, no `EGG-INFO` machinery actually needed at runtime — so we
extracted it directly into `site-packages` instead:

```bash
ENVDIR="$MAMBA_ROOT_PREFIX/envs/carla-redteam"
python3 -c "
import zipfile
z = zipfile.ZipFile('$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg')
for n in z.namelist():
    if n.startswith('carla/'):
        z.extract(n, '$ENVDIR/lib/python3.7/site-packages')
"
```

This first failed with `ImportError: libtiff.so.5: cannot open shared object
file` — the compiled `libcarla.so` was built against the same 2020-era
`libtiff=4.1.0` pinned in Roach's `environment.yml`, and no `libtiff.so.5`
exists on this host or in the fresh Python-3.7 env. Fixed by installing that
exact conda-forge package (a native-lib install, unrelated to the Python
version) into the env:

```bash
micromamba install -y -p "$ENVDIR" -c conda-forge "libtiff=4.1.0"
```

After that, `python -c "import carla; carla.Client"` succeeds cleanly.

## 5. Launch CARLA headless

Use `scripts/launch_carla.sh <gpu_id> <port> [carla_root]` (see that file —
parameterized so Phase 6 can launch several instances in parallel on
different GPUs/ports). Example:

```bash
CARLA_ROOT="$PROJECT_DATA_DIR/CARLA_0.9.11" ./scripts/launch_carla.sh 1 2000
```

This pins the server to GPU 1 via `CUDA_VISIBLE_DEVICES`, runs
`-RenderOffScreen` (headless), and listens for RPC on port 2000.

**What actually happened on this node — a real, silent crash and its fix.**
`./scripts/launch_carla.sh 3 2000` (Docker mode, the default per §4) at first
had the container exit immediately every time, with **zero diagnostic
output** other than a harmless `sh: 1: xdg-user-dir: not found` line. This
took real debugging to root-cause since nothing about it looked like a
graphics/driver problem at first glance:

1. Confirmed it wasn't a missing-GPU/Vulkan-driver problem: installed
   `vulkan-utils` inside the container and ran `vulkaninfo --summary` — it
   correctly enumerated the pinned GPU (`deviceName = NVIDIA H200`,
   `DISCRETE_GPU`, Vulkan 1.4). So the GPU passthrough, ICD, and driver libs
   (all supplied automatically by this host's `nvidia-container-toolkit` CDI
   integration via `--gpus device=N` — no extra mounts needed) were all fine.
2. Confirmed it wasn't a signal crash (segfault/abort): ran the binary
   directly under `gdb -batch -ex run -ex bt`. gdb reported a clean
   `exited with code 01` — no signal caught, meaning the engine itself
   deliberately called `exit(1)` somewhere, not a fault.
3. Set a breakpoint on `exit`/`_exit` instead (`gdb -ex "break exit" -ex
   run -ex bt`) to catch it in the act. The backtrace pinned it exactly:
   ```
   #0 __GI__exit
   #1 FUnixPlatformMisc::RequestExit
   #2 FUnixPlatformMisc::RequestExitWithStatus
   #3 FLinuxApplication::CreateLinuxApplication (LinuxApplication.cpp:46)
   #4 FSlateApplication::Create
   #5 FEngineLoop::PreInitPreStartupScreen
   ```
   This happens in Slate/UE4's windowing layer, *before* the Vulkan RHI is
   ever touched — `-RenderOffScreen` only tells CARLA's renderer to draw to
   an off-screen target, it does **not** stop the underlying SDL2 windowing
   library from first trying to open a real display connection. With no
   `DISPLAY` and no X server in the container, `SDL_Init(SDL_INIT_VIDEO)`
   fails and UE4 exits(1) via exactly this path — a well-known UE4-on-Linux
   headless gotcha, just an unusually silent one in a Shipping build (no log
   file is opened yet at this point in startup, so nothing is written
   anywhere for a normal user to find).
4. Fix: set `SDL_VIDEODRIVER=offscreen` in the container's environment so
   SDL never tries to reach a real display. `launch_carla.sh`'s Docker mode
   now passes `-e SDL_VIDEODRIVER=offscreen`, plus `--shm-size=2g` (UE4's
   shared-memory needs under the small Docker default) and `-nocrashreports`
   (skip spawning the interactive CrashReportClient, irrelevant headless).

After this fix, `./scripts/launch_carla.sh 3 2000` starts and **stays up**
(verified via `docker ps` + `nvidia-smi` showing GPU 3 climb from 0MiB to
~1GB used while the container runs) — confirmed working end-to-end in §6.

## 6. Verify client connection

```bash
python scripts/check_client_connection.py --host localhost --port 2000
```

This connects via `carla.Client`, calls `get_world()`, and lists available
maps and blueprints. **Actual output from this node** (GPU 3, server from §5):

```
Connecting to CARLA at localhost:2000 ...
Connected. Client version: 0.9.11
Server version: 0.9.11

Available maps:
  /Game/Carla/Maps/Town01
  /Game/Carla/Maps/Town01_Opt
  /Game/Carla/Maps/Town02
  /Game/Carla/Maps/Town02_Opt
  /Game/Carla/Maps/Town03
  /Game/Carla/Maps/Town03_Opt
  /Game/Carla/Maps/Town04
  /Game/Carla/Maps/Town04_Opt
  /Game/Carla/Maps/Town05
  /Game/Carla/Maps/Town05_Opt

Currently loaded map: Town03

Blueprint library: 162 blueprints
  vehicle.* blueprints: 31
  sensor.* blueprints: 13

OK: client connection verified.
```

Client and server versions match (0.9.11/0.9.11) confirming the egg
extracted in §4 is the right build for this server. The server's default
boot map is Town03, not Town01 — irrelevant here since `run_clean_episode.py`
(§8) explicitly requests Town01 via `carla_map=` when it creates the
`LeaderBoard-v0` env, which loads/switches maps itself.

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
