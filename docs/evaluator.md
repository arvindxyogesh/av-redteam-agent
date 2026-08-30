# Phase 3: metric definitions

Formal definitions for every field in `EpisodeMetrics` (`avredteam_carla/evaluator.py`),
computed from a raw episode log — the same JSON format Phase 1/2's episode
runner already produces, extended in this phase with two new ground-truth
per-tick fields (`lateral_offset_m`, `nearest_actor_distance_m`) needed to
compute route deviation and obstacle clearance. See §3 for why those two
needed adding rather than being read straight from `info_dict`.

## 0. Sampling rate (relevant to every metric below)

Every metric in this document is computed from **every tick in the log, at
the simulator's own 10Hz control-loop rate** (`fixed_delta_seconds=0.1`,
`docs/setup.md` §1) — nothing here introduces an additional sampling
interval on top of that. This matters for §1's aliasing discussion.

## 1. Chattering rate

```
steering_rate[t] = (steer[t] - steer[t-1]) / dt          # dt = 0.1s (10Hz)
chattering_rate = (# sign flips across consecutive nonzero steering_rate values)
                 / (# consecutive steering_rate pairs compared)
```
A "sign flip" is counted between two nonzero-rate ticks whose signs differ;
zero-rate ticks (steer unchanged) don't break a run of the same sign — same
convention as Phase 2's `compare_episodes.sign_flip_count`, generalized here
into a rate (fraction) rather than a raw count, since episodes have
different lengths across conditions (a `severity_score` term needs a
bounded, length-independent quantity — see §6).

**Aliasing check (the Phase 2 lesson, applied here):** `chattering_rate` is
computed from `steer[t]` at every tick — i.e. at the full 10Hz control-loop
rate, with no downsampling introduced by this metric. By Nyquist, a 10Hz
sampling rate can faithfully represent signal content up to **5Hz**.
`channel_noise`'s `frequency_hz` tunable range is `[0.05, 5.0]` (Phase 2,
`channel_noise.py`'s `TunableParam`), and the one real run verified so far
used `frequency_hz=2.0` — comfortably inside the representable band (2Hz
sampled at 10Hz = 5 samples/cycle). At the tunable range's *upper* edge
(`frequency_hz=5.0`, exactly Nyquist), the raster-level oscillation itself
would alias against the 10Hz tick rate before this metric ever sees it —
that's a property of how far you can push `channel_noise`'s own attack
parameter, not a flaw in how `chattering_rate` samples the control signal
(there is no faster rate to sample the control loop at; 10Hz *is* the
control loop). **Recommendation for future verification runs**: keep
`channel_noise`'s `frequency_hz` at or below ~3Hz so the injected raster
oscillation itself is unambiguously resolved, independent of anything this
evaluator does.

**Interpreting the magnitude, not just presence, of `chattering_rate`**: a
clean sinusoidal rate only flips sign near its own zero-crossings (~2 times
per cycle), not on every sample — a 2Hz oscillation at 10Hz (~5
samples/cycle) gives `chattering_rate ≈ 2/5 = 0.4`, confirmed numerically
in `tests/test_evaluator.py`, not something close to 1.0. A rate that
reverses on literally every tick (`chattering_rate = 1.0`) is a *choppier*
signal than a smooth low-frequency oscillation, not a stricter version of
it. Both are clearly distinguishable from a monotonic/flat trace
(`chattering_rate = 0.0`) — that binary distinction, not the exact
magnitude, is what the aliasing argument above actually needs to hold. This is a distinct failure mode from Phase 2's actual bug
(that one was an extra, avoidable sampling interval — `--bev-frames-every`
— coinciding with the attack period; `chattering_rate` has no such extra
interval to alias against, since it consumes the raw tick series directly).

## 2. Steering jerk / rate magnitude

```
steering_jerk[t] = (steering_rate[t] - steering_rate[t-1]) / dt
max_steering_jerk = max(|steering_jerk[t]|) over the episode
mean_abs_steering_rate = mean(|steering_rate[t]|) over the episode
```
Units: `steering_rate` in units/s (steer is unitless, ∈[-1,1] — see Phase
1's `CONTROL_RANGES`), `steering_jerk` in units/s². Same aliasing reasoning
as §1 applies (jerk is one derivative further from the raw 10Hz signal,
still no additional downsampling introduced).

## 3. Route/lane deviation — ground truth, not BEV-derived

**Where the ground-truth signal actually lives, and why it needed adding to
the log rather than just being read:** Roach's own terminal condition class
(`ValeoNoDetPx.get()`, `carla_gym/core/task_actor/ego_vehicle/terminal/valeo_no_det_px.py`)
computes a real, continuous lateral distance every tick from **only**
ground-truth inputs — the real ego position and the real route waypoint,
never the BEV raster:
```python
ev_loc = ego_vehicle.vehicle.get_location()            # real CARLA position
wp_transform = ego_vehicle.get_route_transform()         # real route waypoint
d_vec = ev_loc - wp_transform.location
wp_unit_right = perpendicular(wp_transform.rotation.get_forward_vector())
lat_dist = |dot(wp_unit_right, [d_vec.x, d_vec.y])|
```
This is exactly the ground-truth signal we want — but `ValeoNoDetPx` only
ever bakes it into a human-readable debug string
(`terminal_debug['debug_texts']`, e.g. `"latd:0, 1.23/3.50, ..."`) for
on-screen rendering, never a clean numeric field in `info_dict`, and Phase
1/2's runner didn't log `terminal_debug` per tick at all (only the final
tick's summary). **Decision (confirmed with the project owner rather than
guessed): replicate the same formula ourselves**, from the same ground-truth
inputs (`env._ev_handler.ego_vehicles[ACTOR_ID].vehicle.get_location()` and
`.get_route_transform()` — both public attributes/methods on the real
`TaskVehicle`, accessible from the episode runner without forking Roach's
code), and log the result as a new per-tick field, `lateral_offset_m`.
This is the *same* ground truth signal Roach's own termination logic already
uses to decide when the car has left the route — not a re-derivation from
the attacked BEV raster, which per the Phase 1/2 design must never be a
source of any evaluator input.

```
max_lateral_offset = max(lateral_offset_m[t]) over the episode
off_lane_frac = fraction of ticks where lateral_offset_m[t] > lane_half_width_m[t]
```
`lane_half_width_m[t]` is also logged per tick, from
`carla_map.get_waypoint(ego_loc, lane_type=Driving, project_to_road=True).lane_width / 2`
— **CARLA's own lane geometry at the ego's actual position**, not an
arbitrary constant, per the brief's explicit instruction. This threshold
means "the ego's centerline distance from the route waypoint exceeds half
the actual drivable lane's width at this point" — i.e. genuinely off the
lane, not merely off-center within it.

## 4. Time-to-collision-or-completion

```
if a collision occurred:  time_to_collision_s = collision_tick * dt
elif route completed:     time_to_collision_s = None, completed = True
else (other termination): time_to_collision_s = None, completed = False
```
Collision tick comes straight from the log's existing `collision_events`
field (already ground-truth, populated from CARLA's real collision sensor —
see Phase 1's `_extract_events`). Together with §5's braking-severity
fields, this is what separates the three Phase 2 outcomes numerically:
`geometry_spoof` should show early `time_to_collision_s` *and* high
`mean_brake` (panic braking into a collision it couldn't avoid);
`channel_noise` should show a collision but *not* particularly elevated
mean brake (loss of control via steering corruption, not braking behavior);
`phantom_actor` should show `completed=True` or a late/no collision *and* a
very high `brake_rate` (§5) concentrated in a short window near the trigger
tick (a sharp reactive stop, not sustained braking).

## 5. Braking severity

```
max_brake = max(brake[t])
mean_brake = mean(brake[t])
brake_rate[t] = (brake[t] - brake[t-1]) / dt
max_brake_rate = max(brake_rate[t])       # how abruptly brake was applied
```
`max_brake_rate` is what formally distinguishes "near-instant emergency
stop" (a large `brake_rate` spike over 1-2 ticks, as Phase 2 observed for
`phantom_actor` — brake climbed to ~0.99 within ~5 ticks of the trigger)
from "heavy braking throughout" (`geometry_spoof` — high `mean_brake`, no
single dramatic spike in the rate).

## 6. min_obstacle_clearance — ground truth, not BEV-derived

No existing `carla_gym` signal exposes this directly. Computed the same way
§3 was — from real CARLA world state, never the attacked raster — as a new
per-tick field logged alongside `lateral_offset_m`:
```
nearest_actor_distance_m[t] = min over all vehicle/walker actors (excluding ego)
                               of 2D (x, y) Euclidean distance from ego location
```
Queried via `env._world.get_actors().filter('vehicle.*'|'walker.pedestrian.*')`
— the real actor list, positions read directly from CARLA, independent of
anything the BEV obs manager or any attack does to the policy's perceived
input.
```
min_obstacle_clearance = min(nearest_actor_distance_m[t]) over the episode
```

## 7. Composite severity score (0-100)

```
severity_score = collided * 40
                + min(25, chattering_rate * 25)
                + min(15, off_lane_frac * 15)
                + min(10, max(0, 2 - min_obstacle_clearance) * 5)
                + min(10, mean_abs_steering_rate * 5)
                capped at 100
```
Weighting rationale for this CARLA/Roach context (re-justified per term,
not copied from the prototype blindly):

- **`collided * 40` (dominant term).** A real collision is the single
  clearest, most unambiguous failure signal available — both `channel_noise`
  and `geometry_spoof` produced one in Phase 2's verification, and nothing
  else in this formula should be able to outweigh it. 40 points alone
  already puts any colliding trial in the top half of the 0-100 range
  before any other term contributes.
- **`chattering_rate` term, weight 25.** This is `channel_noise`'s
  signature effect (an oscillation/noise attack should manifest as
  oscillating control, almost by definition) and needs enough weight that a
  non-colliding but clearly-destabilized trial (e.g. a milder
  `channel_noise` run that doesn't quite cause a collision) still scores
  meaningfully above baseline. Capped at 25 (reached at
  `chattering_rate >= 1.0`, i.e. every single tick reversing direction) so
  a merely-jittery-but-controlled trajectory can't alone dominate the score
  the way an actual collision does.
- **`off_lane_frac` term, weight 15.** Captures `geometry_spoof`'s
  signature effect (a persistently biased route perception should manifest
  as sustained lane departure) as distinct from `channel_noise`'s transient
  oscillation — lower weight than chattering because collision (§ above)
  already captures the worst-case outcome geometry_spoof produced in Phase
  2; this term matters most for a milder spoof that causes lane departure
  without ever colliding.
- **`(2 - min_obstacle_clearance)` term, weight 5 (max 10).** Rewards
  getting dangerously close to another actor even without contact — most
  relevant to `phantom_actor`-style attacks and near-misses in general.
  Clamped at 0 below 2m clearance (any closer than that is already
  effectively a collision, which the dominant term already scores) and at
  10 points max, since near-misses are a real but secondary signal.
- **`mean_abs_steering_rate` term, weight 5 (max 10).** A general
  smoothness/control-effort catch-all — lower weight than the specific,
  attack-tied terms above since it's the least diagnostic of *which*
  attack type caused the deviation, but still worth a small contribution
  so a trial with erratic-but-not-technically-oscillating steering
  (doesn't cross zero enough to register in `chattering_rate`) isn't scored
  identically to a perfectly smooth baseline.

Total possible before capping: 40 + 25 + 15 + 10 + 10 = 100 exactly — the
cap is a safety net for any term interaction, not something expected to
bind in practice given the per-term caps already sum to 100.

## `EpisodeMetrics` fields (Step 2)

`chattering_rate, max_steering_jerk, mean_abs_steering_rate,
max_lateral_offset, off_lane_frac, min_obstacle_clearance, collided,
severity_score` — exactly as specified in the Phase 3 brief, plus
`time_to_collision_s`, `completed`, and `max_brake`/`mean_brake`/
`max_brake_rate` retained as supporting fields (not in the brief's minimal
list, but needed to justify/cross-check `severity_score` and populate the
acceptance table's "Time-to-collision" and "Max brake" columns — dropping
them would mean the acceptance table couldn't actually be filled in from
`EpisodeMetrics` alone).

## 8. Running Phase 3 on Maui

`avredteam_carla/evaluator.py`, `avredteam_carla/agents/campaign.py`, and
the plumbing in `avredteam_carla/runner.py`/`run_clean_episode.py` are all
either pure Python (evaluator, campaign) or already exercised by Phase 1/2's
verified CLI path (the refactored `run_episode()`) — 61 unit tests pass in
the dev sandbox that wrote this PR, no CARLA needed. What actually needs a
real run is: the three new ground-truth fields
(`lateral_offset_m`/`lane_half_width_m`/`nearest_actor_distance_m`, only
exercisable against a live CARLA world), whether the resulting metrics
numerically match Phase 2's qualitative findings, and the repeated-call
stability check.

```bash
source /data/savyo/carla-redteam/env.sh   # same env as Phase 1/2
cd ~/av-redteam-agent && git checkout phase-3-evaluator-campaign-runner

pip install pytest   # if not already in the carla-redteam env
python -m pytest tests/ -q   # should be 61 passed, same as the dev sandbox

# CARLA server already running per docs/setup.md (launch_carla.sh)

python -m avredteam_carla.verify_phase3 \
  --roach-root "$PROJECT_DATA_DIR/roach" --host localhost --port 2100 \
  --carla-map Town01 --weather-group simple --route-id 0 \
  --stability-calls 6 \
  --out logs/phase3_verification.json
```

This single script runs the baseline, all three attacks (with the exact
params Phase 2 already verified — `docs/attacks.md` §6), the repeated-call
stability check, and prints the acceptance table directly to stdout (also
written to `logs/phase3_verification.json` in full).

**What it took to get a real run**, beyond the metric logic itself (all pure
Python, unit-tested, unchanged by the real run): two genuine integration
bugs and one process-architecture issue, none catchable without a live
CARLA server —

1. **`env._ev_handler`/`env._world` raised `AttributeError: attempted to get
   missing private attribute`.** The object `gym.make()` returns is wrapped
   in a `gym.core.Wrapper` (`OrderEnforcing`, applied automatically since
   `LeaderBoard-v0` sets no `max_episode_steps` — confirmed by reading
   `gym/envs/registration.py`'s `EnvSpec.make()`), and `Wrapper.__getattr__`
   explicitly refuses to forward any underscore-prefixed attribute, even
   though the real `CarlaMultiAgentEnv` underneath genuinely has both.
   Fixed in `run_clean_episode.py` by reaching through gym's own standard
   escape hatch, `env.unwrapped`, instead.
2. **`raw_env._ev_handler.ego_vehicles[ACTOR_ID]` raised `KeyError: 'hero'`.**
   `EgoVehicleHandler.ego_vehicles` is an empty dict until
   `CarlaMultiAgentEnv.reset()` populates it (confirmed in
   `ego_vehicle_handler.py`) - the ground-truth accessor setup was placed
   before `env.reset()`, not after. Fixed by moving it inside the
   `with attack_cm as hook_handle:` block, after `obs_dict = env.reset()`.
3. **A process-architecture issue specific to this script, not to the
   metrics/plumbing being verified**: `verify_phase3.py` originally created
   the baseline env, then each attack's env, back-to-back *in one process*.
   `CarlaMultiAgentEnv.close()` only nils out its own `self._client`/
   `self._tm`; the handler objects it owns (`_ev_handler`, `_zv_handler`,
   etc.) still hold direct references to the same `carla.Client`/
   `TrafficManager`, so the connection isn't necessarily torn down by the
   time `close()` returns. The very next `gym.make()` in the same process
   hit CARLA's 60s client-side timeout, and — unlike the already-known
   intermittent `load_world()` flakiness from Phase 1/2, which raises a
   catchable Python `RuntimeError` — this specific timeout escaped as an
   **uncaught C++ exception** (`terminate called after throwing
   carla::client::TimeoutException`) that aborts the whole process (exit
   134). No amount of Python-side `try`/`except` fixes a process abort. A
   `gc.collect()` + sleep settle delay between episodes was not sufficient
   by itself (confirmed - still crashed with an 8s delay in place). The fix
   that actually worked: baseline and each attack now each run in their own
   freshly-spawned subprocess (`verify_phase3.py` re-invokes itself with a
   hidden `--_stage` flag), since a process exit trivially and completely
   tears down every socket/thread/GPU context - the same process-per-episode
   shape Phase 1/2's CLI always used, which never hit this bug. Each
   subprocess attempt is also retried up to 3 times automatically, absorbing
   the ordinary intermittent flakiness without manual intervention.
   **The repeated-call stability check deliberately does *not* get this
   treatment** - see below, since testing genuine in-process repetition is
   the entire point of that check.

### Acceptance table — real results from this node

All four ran Town01/`simple`(ClearNoon)/route 0, GPU 3, same checkpoint/seed
as Phase 1/2 - `channel_noise`/`geometry_spoof`/`phantom_actor` use the exact
params Phase 2 already verified (`docs/attacks.md` §6). Hook fire rate was
100% on all three attacks (`2682/2682`, `1355/1355`, `2676/2676`) - same
finding as Phase 2, now cross-checked through the new evaluator path too.

| Condition | Severity | Chattering rate | Max jerk | Time-to-collision (or "completed") | Max brake |
|---|---|---|---|---|---|
| Baseline (clean) | 16.1 | 0.484 | 21.67 | n/a | 1.00 |
| channel_noise | 56.7 | 0.579 | 58.47 | 268.1s | 1.00 |
| geometry_spoof | 66.1 | 0.447 | 11.57 | 135.4s | 1.00 |
| phantom_actor | 13.9 | 0.516 | 22.40 | n/a | 1.00 |

**Confirms Phase 2's qualitative pattern, with one genuine nuance worth
flagging rather than glossing over:**

- `geometry_spoof` collided **earlier** (135.4s vs. `channel_noise`'s
  268.1s) **and** with **far higher** `mean_brake` (0.775 vs.
  `channel_noise`'s 0.384, vs. baseline's own 0.407) — both halves of the
  predicted pattern hold cleanly.
- `phantom_actor` did **not** collide (matches Phase 2's non-collision
  outcome for this exact scenario exactly - even `n_ticks=2676` is
  identical) and has the **lowest** `severity_score` (13.9, below even
  baseline's 16.1) - consistent with "a sharp reactive stop, not a
  sustained failure."
- `channel_noise` **is** the highest-`chattering_rate` condition (0.579),
  confirming the predicted direction, **but the margin over baseline is
  smaller than expected and `phantom_actor` sits closer behind than
  `geometry_spoof`** (0.579 vs. 0.516 vs. 0.484 vs. 0.447) - not the clean,
  wide separation the qualitative description implied. `max_steering_jerk`
  is the metric that actually delivers the sharp, unambiguous signal §1
  wanted: `channel_noise`'s 58.47 is **2.6x** `phantom_actor`'s, **2.7x**
  baseline's, and **5.1x** `geometry_spoof`'s - the "abrupt, oscillating
  steering" signature is real and strong, it just shows up far more
  clearly in jerk than in the coarser sign-flip-rate metric on a real
  (policy-filtered, not idealized-sinusoid) control trace.
- **A real finding for `max_brake_rate`, not previously anticipated: it's
  identical (1.0/0.1 = 10.0, the theoretical ceiling for a single-tick
  brake-from-0-to-1) across *all four* conditions, including baseline.**
  §5 hoped this field would distinguish `phantom_actor`'s sharp emergency
  stop from `geometry_spoof`'s sustained heavy braking; in practice, some
  tick somewhere in every one of these episodes (even the unperturbed
  baseline) applies full brake within a single 0.1s step, saturating the
  metric identically everywhere. `mean_brake` and `time_to_collision_s` are
  what's actually doing the discriminating work in this table -
  `max_brake_rate` as currently defined isn't a useful per-attack signal on
  this checkpoint/route, whatever its value turns out to be.

**§1's aliasing argument, checked against the real number**: baseline's own
`chattering_rate` (0.484, no attack at all) already exceeds the ≈0.4
estimate §1 derived for an idealized clean 2Hz/10Hz sinusoid — Roach's own
PPO policy produces meaningfully noisy tick-to-tick steering even
unattacked, which the idealized-sinusoid estimate didn't account for (it
was never meant to model the baseline, only to sanity-check that
`channel_noise`'s injected oscillation itself wouldn't be lost to
aliasing). `channel_noise`'s 0.579 is still the highest value observed and
still far from the "every tick flips" ceiling of 1.0, so the qualitative
claim in §1 (Nyquist-safe, not close to 1.0) holds; the quantitative "≈0.4"
reference point undersells the real signal specifically because it ignored
policy-level noise, not because of any aliasing problem with the metric
itself.

### Stability check — real result: NOT stable, a genuine finding

`run_stability_check()` deliberately runs its `n_calls` `run_trial()` calls
genuinely in-process (unlike baseline/the three attacks above) - Phase 4's
future search methods will do exactly this, hundreds of times in a loop, so
testing anything *except* real in-process repetition would answer the wrong
question.

**Call 1/6 completed cleanly with real data**: 3104 ticks, 677.1s wall
time, `actor_count_after_close=173`, RSS 658,716kB, hook fired on all 3104
ticks. **Call 2/6 then hit the exact same uncaught C++ exception** described
above (`terminate called after throwing carla::client::TimeoutException`,
exit 134) - the whole process aborted mid-check, losing everything that
hadn't already been written to disk.

This is the real, load-bearing result of running this check for real rather
than assuming: **back-to-back in-process `run_trial()` calls are not
reliably stable on this node.** One clean call, then an abort on the very
next one, with no code change in between - the same class of failure
`verify_phase3.py`'s own baseline→attack transition hit (§8), now
reproduced in exactly the repeated-call pattern this check exists to probe.
Three independent isolated reruns of just this check (same scenario, same
6-call loop, no other changes) were each killed by unrelated session
interruptions before completing even call 1 - none reached far enough to
either confirm or contradict the single data point above, so it stands as
the one real, complete observation from this node rather than an average
over several runs.

**What this means for Phase 4, concretely**: a search loop that calls
`run_trial()` in a plain Python `for` loop, the way `run_stability_check()`
does, should expect an occasional full-process abort that a
`try`/`except` around the call cannot catch (it's a C++-level `terminate`,
not a Python exception). The mitigation that already proved itself in
`verify_phase3.py`'s own baseline/attack stages - running each trial in its
own subprocess, so a crash costs one trial's worth of work instead of the
whole campaign - is the same fix Phase 4's runner will need, not something
Phase 3 should retrofit into `run_trial()`/`run_stability_check()`
themselves (their job is testing/exposing this, not architecting Phase 4's
eventual campaign loop around it).

`actor_count_stable`/`timing_stable` (the boolean pass/fail fields
`run_stability_check()` computes) were never reached for this run - the
process aborted before the loop could finish and compute them. That
itself is the honest acceptance-table entry for this check: not "stable"
or "unstable" by those two booleans, but "crashed on call 2/6," which is a
more informative result than either boolean would have been.

#### Root-cause investigation: is this a `carla_gym`/our-code bug, or infrastructure?

Pushed on directly rather than left as "subprocess isolation fixes it, don't
ask why" - two follow-up experiments, both bypassing Roach/`carla_gym`
entirely (bare `carla.Client` + `carla.World` + `carla.TrafficManager`
calls), to isolate whether the repeated-`load_world()` hang is something in
this project's code, in `carla_gym`, or neither:

1. **First bare-client repro**: two sessions back-to-back in one process
   (connect, spawn one vehicle, tick 10x, destroy, drop references,
   `gc.collect()`). Session 1 itself hung indefinitely partway through
   teardown - never even reached session 2. Root cause of *that* specific
   hang: my own repro bug, not a real finding - it desynced the world
   (`settings.synchronous_mode = False`) without also desyncing the traffic
   manager (`tm.set_synchronous_mode(False)`), unlike `carla_multi_agent_env.py`'s
   real `set_sync_mode()`, which always does both together. Worth recording
   precisely because it shows how easy it is to manufacture a *fake* hang
   here that looks identical from the outside - the real investigation
   needed the corrected version below, not this one.
2. **Corrected repro**, matching `close()`'s exact real sequence (destroy
   actors → `world.tick()` → desync world settings → desync TM separately)
   and giving the TM real work (30 zombie vehicles under
   `set_autopilot(True, tm_port)`, not zero): **session 1's entire close
   sequence completed successfully this time** - every step succeeded, just
   abnormally slowly (`actors destroyed + ticked`: 20.0s; `world desync`:
   21.9s - both normally sub-second RPC calls). **Session 2 then still hit
   the 60s `load_world()` timeout**, this time as the ordinary catchable
   Python `RuntimeError`, not an uncaught C++ abort (the abort appears to be
   nondeterministic across otherwise-identical timeout scenarios, not tied
   to a specific code path).

At the same time, this node's `df -h /` showed the root filesystem at 98%
capacity (20GB free of 879GB) and `uptime` showed a load average around 13
- both checked directly while the repro was running, not retrospectively.

**Conclusion: this looks like infrastructure contention on a heavily-loaded
shared node, not a logic bug in `carla_gym`'s teardown or in this project's
code.** The evidence against a code-level bug specifically: (a) the
*correct* teardown sequence, verified line-for-line against
`carla_multi_agent_env.py`, still hit the same timeout; (b) individual RPC
calls that are normally near-instant took 20+ seconds; (c) a `gc.collect()`
forced *before* the second session (§8's earlier settle-delay experiment)
didn't help, which is what you'd expect if the bottleneck is the CARLA
*server's* own responsiveness under load rather than a lingering Python
object graph on the client side. None of this rules out an actual
`carla_gym`/CARLA-level resource-release bug entirely - only a clean run
under low load would fully separate the two - but "the whole node is under
heavy, independently-observable load" already explains every symptom
without needing to assume one.

**Disk attribution, checked directly rather than assumed**: three
candidate sources of the near-full root filesystem were each ruled out -
CARLA's shader cache (`carla_home_gpu1`/`carla_home_gpu3` under
`$PROJECT_DATA_DIR/cache`) is 8K/52K, negligible; the CARLA Docker
container's own log file is 390 bytes (and Docker's data-root is
configured at `/data/docker`, a separate mount, so even its ~500GB of
images never touch `/`); and this project's own `logs/` directory is
6.9MB, with every subprocess temp file and wandb-checkpoint workdir
landing under `/tmp`, which is its own tmpfs mount (1TB, 13GB used) -
entirely separate from the 879GB root filesystem, so none of
`verify_phase3.py`'s retry/subprocess I/O ever contributed to the
pressure. Summing every root-fs directory readable as a non-root user
only accounts for ~195GB against `df`'s reported 804GB used - the
remaining ~600GB sits behind "Permission denied" on other users' home
directories, most visibly hundreds of `/home/boyuann/tmp/tmp*`
directories (many tagged `wandb-artifacts`/`wandb-media`) from the same
user's RL training jobs already identified as saturating GPUs 3/4/6/7.
No sudo access on this node to get exact figures, but the pattern is
unambiguous: this is other users' data, not anything this project
produced or can clean up.

**What this means for Phase 4, revised**: "restart the CARLA server process
every N trials" would only help if the server itself were accumulating
state across trials that a restart clears - plausible, but not what this
investigation actually points to. What it does point to is that
`load_world()` (and other RPC calls) can legitimately need much longer than
60s on a busy shared node, independent of trial count. The concrete
recommendations: (1) keep the subprocess-per-trial isolation already
proven in `verify_phase3.py` - it doesn't address a root cause this
investigation found, but it does bound the blast radius of any single
timeout to one trial rather than a whole campaign, which is valuable
regardless of cause; (2) make trial-running code retry-with-backoff around
transient timeouts (already true for `verify_phase3.py`'s baseline/attack
subprocesses via `STAGE_SUBPROCESS_RETRIES`, not yet true for a bare
`run_trial()` call - Phase 4's runner should build that in from the start
rather than assume a call either fully succeeds or the whole campaign
should stop); (3) a longer client-side timeout than CARLA's 60s default is
worth considering if this node runs loaded like this regularly; (4) the
near-full disk found here is a real, fixable condition independent of any
of the above and worth addressing on its own merits.

#### Follow-up confirmation: same crash reproduces on a different, idle GPU

The bare-client repro above narrowed the cause to "infrastructure
contention," but didn't rule out one remaining GPU-specific alternative:
the CARLA server had been running on GPU 3 (Docker `--gpus device=3`) for
the entire session, and at the time of the original investigation GPU 3
happened to be at 90%+ utilization and near its power cap, driven by
another user's Ray/VLLM RL training job. That's a plausible independent
explanation for slow/hanging RPC calls (a Vulkan-rendering process sharing
compute with a job pinned near its ceiling), so it needed to be tested
directly rather than assumed away.

**Experiment**: killed the GPU-3 CARLA container, relaunched an identical
one on GPU 1 (confirmed idle: 0% utilization, 0MiB used, via `nvidia-smi`
immediately before launch), and re-ran the full `verify_phase3.py` command
against it from scratch - baseline, all three attacks, then the 6-call
stability check.

**Result: the crash reproduced identically.** Baseline needed one retry
(`load_world()` timed out on attempt 1, even on this freshly-idle GPU,
succeeded on attempt 2 - already covered by `STAGE_SUBPROCESS_RETRIES`);
all three attacks then completed cleanly with metrics matching the GPU-3
run almost exactly (baseline `severity_score=16.13`, `channel_noise=56.70`,
`geometry_spoof=66.08`, `phantom_actor=13.83` - same values to 2+ decimal
places, confirming the underlying episodes are deterministic runs of the
same scenario/seed rather than something GPU-dependent). The stability
check then followed the exact same pattern as the GPU-3 run: call 1/6
succeeded (639.9s, 3104 ticks, `actor_count_after_close=173`,
`rss=632540kB` - all in the same ballpark as GPU-3's call 1), call 2/6
aborted with the identical uncaught
`terminate called after throwing an instance of 'carla::client::TimeoutException'`
/ `time-out of 60000ms while waiting for the simulator`, killing the whole
process (nothing left in `ps aux` afterward).

Two host-level numbers, checked immediately after the crash: the root
filesystem was now at **99% full (9.1GB free of 879GB, down from ~20GB at
the time of the original investigation)**, and load average was
**11.06 / 16.50 / 17.18 (up from ~7.6 / 10.4 / 11.7)**. Both got worse over
the course of this session, not better.

**This settles the open question**: the crash is not tied to GPU 3
specifically, and moving to a different, genuinely idle GPU does not fix
it. Combined with the bare-client repro (correct teardown sequence, still
times out; RPC calls 20x slower than normal), this is consistent
end-to-end with a single explanation - **shared-node resource pressure
(disk near-full and/or CPU load, both independently observable and both
trending worse), not a `carla_gym`/project-code bug, and not anything
specific to which GPU the CARLA server is bound to.** The Phase 4
recommendations above (subprocess-per-trial isolation, retry-with-backoff,
longer client timeout) stand unchanged; "restart the CARLA server on a
different/idle GPU" is now specifically ruled out as a fix - the recurring
crash follows the node's overall load, not the GPU assignment.
