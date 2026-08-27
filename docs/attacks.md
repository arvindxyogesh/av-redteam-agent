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

**Not yet confirmed against a real run** (do this first when back on Maui,
before trusting attack results):
- The actual `input_states` list from the checkpoint's downloaded
  `config_agent.yaml` (§3) — confirm it's `[control, vel_xy]` as assumed, or
  update the state-vector layout above if not.
- That monkeypatching `RlBirdviewWrapper.process_obs` at the point
  `run_clean_episode.py` imports it actually takes effect on the *instance*
  Roach's own `RlBirdviewAgent` calls (should work since it's a staticmethod
  looked up on the class at call time, not bound early — but confirm with a
  trivial "does control change at all" smoke test before trusting any
  specific attack's numbers).
