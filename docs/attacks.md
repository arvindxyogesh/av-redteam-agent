# Phase 2: Roach's actual observation space (read from source, not guessed)

Per the Phase 2 brief, this document is written entirely from reading the real
`carla-roach` source (the same checkout used in Phase 1), not from the paper.
Exact files and line numbers are cited so this can be re-verified.

Source files referenced (paths relative to the `carla-roach` repo root):
- `carla_gym/core/obs_manager/birdview/chauffeurnet.py` — builds the BEV tensor
- `agents/rl_birdview/utils/rl_birdview_wrapper.py` — converts raw env obs into
  exactly what the policy network receives
- `agents/rl_birdview/rl_birdview_agent.py` — calls the above, then the policy
- `carla_gym/core/obs_manager/actor_state/{control,velocity,speed}.py` — scalar
  state fields

## 1. There are two BEV outputs — only one reaches the policy

`ObsManager.get_observation()` in `chauffeurnet.py` (line 207) returns
```python
obs_dict = {'rendered': image, 'masks': masks}
```
**`rendered`** — `(192, 192, 3)` uint8 RGB, human-readable colorized picture
(road=grey, route=light grey, lanes=magenta, vehicles=blue, walkers=cyan,
traffic lights/stops tinted green/yellow/red, ego=white). This is **only**
used for logging/video (`agent.render()`, `im_render`) — **the policy network
never sees it.** Don't attack this thinking it affects control; it's cosmetic.

**`masks`** — this is the actual policy input. `(15, 192, 192)` uint8,
**channels-first**. This is what every attack in this phase must perturb.

## 2. BEV raster (`masks`) — exact shape, channels, values

- Shape: `(C, H, W) = (15, 192, 192)`, dtype `uint8`
- `H = W = 192` (`width_in_pixels` in `config/agent/ppo/obs_configs/birdview.yaml`)
- `pixels_per_meter = 5.0` → each pixel = 20cm; the raster covers a
  192/5 = 38.4m square around the ego
- Ego position within the raster is fixed: `pixels_ev_to_bottom = 40px = 8m`
  from the bottom edge, horizontally centered — i.e. the ego is not at the
  raster's center, it's offset toward the bottom (more raster area ahead of
  the car than behind, since forward context matters more for driving)
- `_masks_channels = 3 + 3*len(history_idx)`. With
  `history_idx: [-16, -11, -6, -1]` (4 entries, ticks relative to now at
  10Hz sim → -16 = 1.6s ago ... -1 = last tick), that's `3 + 3*4 = 15`.

Channel index → content (built at `chauffeurnet.py:204`,
`masks = np.stack((c_road, c_route, c_lane, *c_vehicle_history, *c_walker_history, *c_tl_history), axis=2)`
then transposed to channels-first):

| Ch. | Content | Values |
|---|---|---|
| 0 | `road` — static road-surface mask for the current map (loaded from a per-town `.h5` file, warped into ego frame) | `{0, 255}` |
| 1 | `route` — the planned route polyline, drawn as a 16px-thick line through the next 80 waypoints (`route_plan[0:80]`) | `{0, 255}` |
| 2 | `lane` — lane markings; `255` = solid/all lane marking, `120` = white broken lane marking overlaid on top | `{0, 120, 255}` |
| 3–6 | `vehicle` history, oldest→newest: t-16, t-11, t-6, **t-1 (channel 6 = most recent/"current")** | `{0, 255}` per channel |
| 7–10 | `walker` (pedestrian) history, same t-16→t-1 order, **channel 10 = most recent** | `{0, 255}` per channel |
| 11–14 | `traffic light / stop sign` composite history, same order, **channel 14 = most recent**. Each channel encodes green=80, yellow=170, red-or-stop=255 (later draws in that order overwrite earlier, so red/stop wins on overlap) | `{0, 80, 170, 255}` |

Notes:
- Vehicle/walker/tl channels are **history**, not one-per-object-type-only —
  attacking "the vehicle channel" for a phantom-actor injection should target
  the **most recent** slice (channel 6 for vehicles, channel 10 for walkers)
  since that's the one most directly driving the current-tick decision, though
  a persistent phantom should be injected across all 4 history slices to avoid
  looking like a single-frame sensor glitch the policy might discount.
- `road` and `lane` are static per-map, loaded from HDF5 — perturbing these
  doesn't change with ego motion except via the warp transform, so a
  "geometry spoof" attack (biasing the route/lane channel) needs to apply its
  offset in the same *warped/pixel* frame the mask already occupies, not in
  world coordinates.
- The route channel (1) is the natural "steering target" channel — biasing it
  sideways is the most direct way to fake a lane/heading error, and is what
  Step 2's geometry-spoof attack targets by default.

## 3. Scalar state vector — exact fields, order, units, range

Built in `RlBirdviewWrapper.process_obs()` (`rl_birdview_wrapper.py:160`),
concatenated **in this exact order**, each block only included if its name is
in `input_states`:

| Block (if in `input_states`) | Fields appended, in order | Units / range |
|---|---|---|
| `speed` | `speed_xy` (1 val) | m/s, declared range [-10, 30] (see `speed.py`) |
| `speed_limit` | `speed_limit` (1 val) | m/s (`get_speed_limit()/3.6*0.8`), range [0, 50] |
| `control` | `throttle`, `steer`, `brake`, `gear/5.0` (4 vals) | throttle∈[0,1], steer∈[-1,1], brake∈[0,1], gear normalized to ∈[0,1] (raw gear 0-5) |
| `acc_xy` | `acc_x`, `acc_y` (2 vals, **ego-frame**, via `vec_global_to_ref`) | m/s², declared range [-1000,1000] |
| `vel_xy` | `vel_x`, `vel_y` (2 vals, **ego-frame**) | m/s, declared range [-100,100] |
| `vel_ang_z` | `vel_ang_z` (1 val, **world-frame** — not rotated) | rad/s, declared range [-1000,1000] |

**Which blocks Roach's actual checkpoint uses is config-driven, not hardcoded
in this file** — the repo's default (`config/agent/ppo.yaml`,
`env_wrapper.kwargs.input_states: [control, vel_xy]`) gives a 6-float vector
`[throttle, steer, brake, gear/5.0, vel_x, vel_y]`, but the *authoritative*
source is the `input_states` list inside the checkpoint's own
`config_agent.yaml`, downloaded from W&B at runtime (see Phase 1's
`run_clean_episode.py`, which already logs this). **Confirm the exact
`input_states` list from that downloaded file on Maui before assuming 6
floats in this exact order** — if it differs, update this table and every
attack's `scalar_state` indexing accordingly. This is flagged rather than
assumed per the Phase 2 brief's instruction not to guess.

At eval time (`train=False`), both tensors get a batch dimension prepended:
`birdview` → `(1, 15, 192, 192)` uint8, `state` → `(1, N)` float32.

## 4. The exact interception point

`RlBirdviewAgent.run_step()` (`rl_birdview_agent.py`):
```python
policy_input = self._wrapper_class.process_obs(
    input_data, self._wrapper_kwargs['input_states'], train=False)
actions, values, log_probs, mu, sigma, features = self._policy.forward(
    policy_input, deterministic=True, clip_action=True)
```
`self._wrapper_class` is `agents.rl_birdview.utils.rl_birdview_wrapper.RlBirdviewWrapper`,
loaded dynamically via `load_entry_point()` — not imported directly by name.
`process_obs` is a `@staticmethod` on that class.

**This is the interception point for every attack in this phase**: everything
before it (`input_data`, straight from `carla_gym`'s `ObsManagerHandler`) is
ground truth and must stay untouched; everything after
`policy_input = ...process_obs(...)` and before `self._policy.forward(...)`
is what the policy actually acts on. An attack transforms
`policy_input['birdview']` and/or `policy_input['state']` in that gap.

Because `process_obs` is referenced as `self._wrapper_class.process_obs` (the
class is resolved dynamically at agent-load time, not imported by name
anywhere we control), the integration in this phase monkeypatches
`RlBirdviewWrapper.process_obs` itself at runtime — see
`avredteam_carla/attacks/hook.py`. This avoids forking or editing Roach's
vendored code, matching Phase 1's approach.

## 5. What's confirmed vs. what needs a Maui run to verify

Confirmed by reading source (this document): BEV channel layout and values,
`rendered` vs `masks` distinction, the `process_obs` interception point, the
scalar-state block structure and per-block units/order.

**Confirmed against a real Maui run** (both of §5's open questions from the
dev-sandbox version of this doc):
- §3's `input_states` assumption is **exactly correct**. The actual
  checkpoint's downloaded `config_agent.yaml` (`ckpt_11833344.pth`, run
  `iccv21-roach/trained-models/1929isj0`) has
  `env_wrapper.kwargs.input_states: [control, vel_xy]`, confirming the
  6-float `[throttle, steer, brake, gear/5.0, vel_x, vel_y]` layout §3
  already assumed. Its `obs_configs.birdview` block also independently
  confirms every BEV layout constant in §2 (`width_in_pixels: 192`,
  `pixels_ev_to_bottom: 40`, `pixels_per_meter: 5.0`,
  `history_idx: [-16, -11, -6, -1]`). No corrections needed to §2/§3.
- The `RlBirdviewWrapper.process_obs` monkeypatch **does take effect** on
  the real agent, with a 100% fire rate: every one of the three attacks
  below fired on every single tick of its episode (e.g. `2682/2682`,
  `1355/1355`, `2676/2676`) — `ticks_patched` never came in below `n_ticks`
  across any run in this phase, so `install_attack()`'s reasoning in
  `hook.py` (patch the class attribute, since `self._wrapper_class` is the
  class object itself, looked up fresh on every call) holds up in practice,
  not just in theory.

## 6. Running Phase 2 on Maui

Everything under `avredteam_carla/attacks/` and its unit tests
(`tests/test_attacks.py`, `tests/test_visualize.py`,
`tests/test_compare_episodes.py`) is pure Python/numpy and was already run
and passing in the dev sandbox that wrote this PR — 40 tests, no CARLA
needed. Re-running them on Maui is still worth doing once (confirms the
Roach conda env's numpy/pytest work the same way), but the part that
actually needs verifying here is the CARLA/Roach integration:

```bash
source /data/savyo/carla-redteam/env.sh   # same env as Phase 1
cd ~/av-redteam-agent && git checkout phase-2-bev-attack-library

pip install pytest   # if not already in the carla-redteam env
python -m pytest tests/ -q   # should be 40 passed, same as the dev sandbox

# CARLA server already running per docs/setup.md (launch_carla.sh)

# 1. Baseline clean episode (Phase 1 behavior, unchanged)
python -m avredteam_carla.run_clean_episode \
  --roach-root "$PROJECT_DATA_DIR/roach" --host localhost --port 2100 \
  --carla-map Town01 --weather-group simple --route-id 0 \
  --out logs/phase2_clean.json

# 2. One attacked episode per attack type
python -m avredteam_carla.run_clean_episode \
  --roach-root "$PROJECT_DATA_DIR/roach" --host localhost --port 2100 \
  --carla-map Town01 --weather-group simple --route-id 0 \
  --attack channel_noise --attack-param channel=1 --attack-param amplitude=100 --attack-param frequency_hz=2 \
  --bev-frames-every 100 \
  --out logs/phase2_channel_noise.json

python -m avredteam_carla.run_clean_episode \
  --roach-root "$PROJECT_DATA_DIR/roach" --host localhost --port 2100 \
  --carla-map Town01 --weather-group simple --route-id 0 \
  --attack geometry_spoof --attack-param max_offset_m=3 --attack-param ramp_ticks=30 \
  --bev-frames-every 100 \
  --out logs/phase2_geometry_spoof.json

python -m avredteam_carla.run_clean_episode \
  --roach-root "$PROJECT_DATA_DIR/roach" --host localhost --port 2100 \
  --carla-map Town01 --weather-group simple --route-id 0 \
  --attack phantom_actor --attack-param distance_m=15 --attack-param trigger_tick=50 \
  --bev-frames-every 100 \
  --out logs/phase2_phantom_actor.json

# 3. Compare each attacked run against the clean baseline
python -m avredteam_carla.compare_episodes --clean logs/phase2_clean.json --attacked logs/phase2_channel_noise.json
python -m avredteam_carla.compare_episodes --clean logs/phase2_clean.json --attacked logs/phase2_geometry_spoof.json
python -m avredteam_carla.compare_episodes --clean logs/phase2_clean.json --attacked logs/phase2_phantom_actor.json
```

For each run, check the log line `Attack hook fired on N/M ticks` — if
`ticks_patched=0`, the monkeypatch didn't take effect and nothing else in
that run is trustworthy; stop and re-open §5's open question before
proceeding, don't just report the (meaningless) numbers.

**Operational note from this Maui run:** the same intermittent
`client.load_world()`/`get_trafficmanager()` 60s-RPC-timeout seen in Phase 1
(docs/setup.md §7) recurred here too, roughly every other fresh episode —
each `run_clean_episode.py` invocation creates a brand-new `LeaderboardEnv`
(hence a fresh `load_world()` call) even against an already-running, already
map-loaded CARLA container. A same-container retry succeeded every time it
happened; no code fix was needed, this is just cold-RPC flakiness inherent
to spinning up 120+120 zombie actors fresh each episode. Budget for 1-2
retries per episode when scripting this.

### Results, per attack (real numbers from this Maui run)

All three ran against the same clean baseline
(`logs/phase2_clean.json`: Town01/`simple`/route 0, 3998 ticks,
`vehicle_blocked`, deterministic — identical to Phase 1's result, as
expected for a zero-perturbation rerun of the same seed/checkpoint/route).

**`channel_noise`** (`channel=1` [route], `amplitude=100`, `frequency_hz=2`):
- Hook fired **2682/2682** ticks (100%).
- Ran start-to-finish, no crash, **0 NaN ticks**, all controls in range.
- Episode ended in a real **collision** (`collision_type=1`, vehicle) at
  tick 2682 — the clean baseline never collides (ends via `vehicle_blocked`
  at the route's end). `compare_episodes.py`: `steer_sign_flips` 1947→1552,
  `mean_abs_steer_diff`=0.065 (paired, over the 2682 overlapping ticks),
  `mean_speed` actually *rose* slightly (1.92→2.86 m/s) rather than
  dropping — a corrupted route channel didn't make the car more cautious,
  it drove into another vehicle.
- **Visual check found a real gotcha, not a failure of the attack**: every
  saved PNG pair (`logs/bev_frames/channel_noise/tick_*.png`, the default
  `--bev-frames-every 100`) shows **zero pixel difference** between clean
  and attacked. This isn't the attack failing — it's exact aliasing between
  the sampling interval and the attack's own period: at `frequency_hz=2` and
  `SIM_HZ=10` (fixed, see `_util.py`), one full oscillation is exactly 5
  ticks, and `100 % 5 == 0`, so *every* tick sampled by `--bev-frames-every
  100` lands on the same zero-crossing phase (confirmed numerically:
  `amplitude * sin(2*pi*freq*(tick/SIM_HZ))` computes to `~0.00` at ticks 0,
  100, 200, ... 2600). A supplementary short run (90 ticks,
  `--bev-frames-every 17` — coprime with the 5-tick period,
  `logs/phase2_channel_noise_visualcheck.json` /
  `logs/bev_frames_aliasing_check/channel_noise/`) confirms the attack
  really is perturbing the channel correctly: pixel diffs are ~12.3M at
  ticks with a large *positive* offset (e.g. tick 17, offset≈+59) and
  exactly 0 at ticks with a *negative* offset (e.g. tick 34, offset≈-95).
  That asymmetry is expected, not a second bug — `visualize.py` colors a
  pixel by a boolean `> 0` threshold, so a positive offset (background
  0→nonzero) floods the whole raster with route-color and is dramatically
  visible, while a negative offset (foreground 255→still-nonzero after
  clipping) doesn't cross that threshold and is invisible under this
  particular renderer regardless of sampling. At tick 17 the attacked frame
  shows the *entire* visible raster flooded with route-gray, completely
  destroying the road/lane/route distinction — a clear, unambiguous visual
  confirmation once sampled at a non-aliased phase. **Takeaway for future
  runs: don't use a `--bev-frames-every` that's a multiple of
  `SIM_HZ / frequency_hz` for this attack**, or the sanity-check PNGs will
  silently show nothing despite the attack working correctly.

**`geometry_spoof`** (`channel=1` [route], `max_offset_m=3`,
`ramp_ticks=30`):
- Hook fired **1355/1355** ticks (100%).
- Ran start-to-finish, no crash, 0 NaN ticks, all controls in range.
- Episode ended in a real **collision** at tick 1355 — much earlier than
  clean's 3998-tick `vehicle_blocked` ending. `compare_episodes.py`:
  `steer_sign_flips` 1947→606 (far fewer reversals — a persistently biased
  route signal produces a persistent, not oscillating, steering error),
  `mean_abs_steer_diff`=0.068, `mean_brake` 0.41→0.77 and `mean_speed`
  1.92→0.40 — the car braked hard and crawled, consistent with perceiving a
  route that no longer matches the drivable lane.
- Visual check (no aliasing risk here — a ramp-then-hold, not periodic):
  every sampled pair from tick 100 onward shows a large, consistent
  pixel diff (~750K-860K). Eyeballing `tick_000100_{clean,attacked}.png`
  directly: the route polyline is visibly shifted right relative to the
  lane it should be centered in — exactly the intended effect.

**`phantom_actor`** (`distance_m=15`, `trigger_tick=50`, vehicle blob,
default `lateral_offset_m=0`/`blob_radius_m=1`):
- Hook fired **2676/2676** ticks (100%).
- Ran start-to-finish, no crash, 0 NaN ticks, all controls in range.
- **Immediate, dramatic reaction right at the trigger tick**: throttle
  briefly spikes to 1.0 at tick 50-51 then brake climbs to ~0.99 by tick 55,
  with ground-truth speed dropping from ~0.2 m/s to ~0.0 m/s within 7 ticks
  of the phantom appearing — the clearest single before/after signal of any
  attack in this phase, visible directly in the raw tick data without
  needing `compare_episodes.py`. `compare_episodes.py`: `mean_brake`
  0.41→0.29 and `mean_speed` 1.92→2.88 overall (the initial hard stop is a
  small fraction of a 2676-tick episode; the car resumes cruising once the
  perpetually-15m-ahead phantom stops reading as an imminent threat -
  because `apply()` recomputes the phantom at a fixed *ego-relative*
  distance every tick, it never actually gets closer or farther as the ego
  moves, which is itself worth flagging as a modeling choice: this attack
  simulates "there is always a car exactly 15m ahead," not "a car appeared
  once at a fixed world location").
- **Termination reason came back `unknown`** — this is real, and is exactly
  the one gap §5/`_termination_reason()`'s own docstring already flagged:
  none of `collisions_*`, `red_light`, `stop_infraction`, `timeout`, or
  `vehicle_blocked` fired (`final_episode_event` confirms all empty/zero),
  `is_route_completed` is `0.0` despite `route_completed_in_km` (0.760km)
  slightly exceeding `route_length_in_km` (0.748km) — consistent with
  `ValeoNoDetPx`'s own inline lateral-distance-from-route condition ending
  the episode, which (per `terminal/valeo_no_det_px.py`) is computed
  in-line and never written to any criterion buffer this project's
  classifier can read. Not a bug in this phase's code; a pre-documented
  blind spot in what's observable from the log alone.
- Visual check: pixel diff is exactly 0 for the tick-0 frame (before
  `trigger_tick=50`) and a consistent ~26K-32K for every frame after.
  `tick_000100_{clean,attacked}.png` shows a distinct blue vehicle-colored
  disk directly ahead of the ego on the route polyline — unmistakable.

### `compare_episodes.py` output files

`logs/compare_channel_noise.json`, `logs/compare_geometry_spoof.json`,
`logs/compare_phantom_actor.json` hold the full metrics dict for each
comparison (not just the headline numbers pulled out above).

## 7. Acceptance criteria — real results from this node

| Criterion | Result |
|---|---|
| §3's `input_states` assumption confirmed against the real checkpoint config | **PASS.** `[control, vel_xy]`, exactly as assumed — see §5. |
| Monkeypatch takes effect on the real `RlBirdviewAgent` | **PASS.** 100% hook-fire rate (`ticks_patched == n_ticks`) on all three attacked episodes — `2682/2682`, `1355/1355`, `2676/2676`. |
| Each attacked episode runs start-to-finish, no crash/NaN | **PASS**, all three. `channel_noise` and `geometry_spoof` ended via a real collision; `phantom_actor` via an unclassified-but-real terminal condition (see write-up above) — none via a script error. |
| Each attack measurably changes Roach's behavior vs. clean baseline | **PASS**, all three, each via a different real mechanism: `channel_noise` → later collision + speed increase; `geometry_spoof` → much earlier collision + heavy braking/crawling; `phantom_actor` → an immediate, sharp emergency-stop reaction within ~7 ticks of onset. Full metrics in `logs/compare_*.json`. |
| Saved BEV PNGs visually confirm each attack's expected effect | **PASS for `geometry_spoof` and `phantom_actor`** at the documented `--bev-frames-every 100` sampling. **Initially inconclusive for `channel_noise`** at that same sampling — root-caused to a real aliasing artifact (100 is an exact multiple of the attack's 5-tick period) rather than the attack not working; **confirmed PASS** via a supplementary non-aliased sampling run (`--bev-frames-every 17`), which shows dramatic, unambiguous flooding of the route channel at peak-offset ticks. |

Everything in this table is a real, observed result from this node, not an
expectation.
