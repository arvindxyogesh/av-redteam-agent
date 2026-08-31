# Phase 4: search methods (random search, Bayesian optimization, LLM agent)

Following Phase 1-3 (PRs #1-3, unmerged as of this phase - see TASK.md).
This phase adds three search methods that all call `run_trial()` in a
loop, sharing one interface (`avredteam_carla.agents.search.SearchMethod`),
so they're directly comparable on the same scenario/budget/seed. It also
resolves two things the Phase 4 brief left open pending verification
(Step 0's length-normalization question) or turned out to be moot once
checked against the real code (the `max_brake_rate` composite-formula
"fix").

**What could and couldn't be run in this dev sandbox**: no live CARLA
server, no GPU, and none of Phase 3's real per-tick logs (gitignored,
live only on the Maui cluster's filesystem - see `.gitignore`). Everything
below that needs those is written and ready but unverified against real
hardware; everywhere that's true is called out explicitly rather than
presenting sandbox unit tests as if they were a real run. 137 unit tests
pass in the dev sandbox that wrote this PR (up from Phase 3's 61) - pure
Python/stdlib plus `optuna` and `anthropic`, no CARLA needed for any of
them, matching Phase 1-3's own pattern of what a dev sandbox can and can't
verify.

## 0. Step 0 — length-normalization: a real finding, and a real gap

**Establishable without real hardware, and established**: checked
`avredteam_carla/evaluator.py`'s actual code (not assumed) for whether any
`EpisodeMetrics` field carries a length bias. `max_lateral_offset` and
`min_obstacle_clearance` are a running max/min taken over the *entire*
episode's tick series (`docs/evaluator.md` #3/#6). That makes their
length-sensitivity a mathematical certainty, not something real data is
needed to discover: a min taken over a longer prefix of the same series
can only stay equal or decrease as more ticks are added (a min over a
superset is `<=` a min over any subset it contains), and symmetrically a
max can only stay equal or increase. `min_obstacle_clearance` feeds
directly into `severity_score`'s `(2 - clearance) * 5` term
(`docs/evaluator.md` #7), so this is a structural length-sensitivity in
the composite score itself, not just a curiosity about a supporting
field. `off_lane_frac` and `chattering_rate`, by contrast, are already
rate/fraction-normalized (a count divided by `n_ticks` or by the number of
consecutive-tick pairs compared), so they don't carry this same bias -
more ticks changes their *precision*, not their *expected direction*.

Locked in with unit tests on synthetic random-walk logs
(`tests/test_analyze_episode_length_bias.py`) - the input series are
*not* constructed to be monotonic, so the monotonicity the tests confirm
comes from `evaluate()`'s own min/max, not from a rigged input.
`avredteam_carla/analyze_episode_length_bias.py` is the reusable tool:
`analyze_prefix_length_sensitivity(log)` truncates one log into
increasingly long prefixes and reports each candidate metric's Pearson
correlation with `n_ticks` within that one log - which isolates episode
length as the only varying factor, unlike a raw cross-condition
comparison (see below).

**NOT establishable without real per-tick logs, and not fabricated
here**: how *large* this bias actually is on Phase 3's real runs. Phase
3's four real logs spanned genuinely different tick counts (1355-3104,
`docs/evaluator.md`'s acceptance table) *because* they represent
different attacks - a raw length-vs-metric correlation across just those
four points can't separate "this attack caused both a different length
and a different severity" from "length alone moved the metric." The
prefix-truncation trick above sidesteps that confound but still needs at
least one real log file to run against, and none exists in this sandbox.

**Left undecided, per Step 0's own instruction not to fix a problem
before verification confirms its real magnitude**: `severity_score`'s
formula is unchanged in this PR. Two candidate normalizations worth
testing once real magnitude data exists (not applied, not yet chosen):
- Replace the hard `min`/`max` with something less sensitive to a single
  extreme tick - e.g. a low/high percentile (5th/95th) of the per-tick
  series, or the mean of the worst-K ticks.
- Compute the extrema over a fixed time window (e.g. the first 60s) rather
  than the whole variable-length episode.

**Concrete next step for a real-hardware session**: run
`python -m avredteam_carla.analyze_episode_length_bias --log <baseline.json> --log <channel_noise.json> --log <geometry_spoof.json> --log <phantom_actor.json> --out logs/phase4_length_bias.json`
against Phase 3's real logs (rerun `verify_phase3.py` first if they
weren't kept on disk), read off the real `min_obstacle_clearance`/
`max_lateral_offset` correlation-with-length numbers from the printed
table, and only then decide whether the magnitude is large enough to
justify one of the two candidate fixes above (or a different one) -
getting sign-off before applying it project-wide, per the original
instruction, since a wrong fix here propagates through Phases 5-6.

## 1. `max_brake_rate` "fix" — checked, turned out to be a no-op

The Phase 4 brief instructed dropping `max_brake_rate` from
`severity_score`'s composite formula, citing the real Phase 3 finding that
it saturates identically (10.0) across all four conditions including
baseline. Checked directly against the real formula
(`avredteam_carla/evaluator.py`) before touching anything: `max_brake_rate`
was never a term there - the five real terms (`collided`, `chattering_rate`,
`off_lane_frac`, `min_obstacle_clearance`, `mean_abs_steering_rate`) already
summed to exactly 100 before capping (`docs/evaluator.md` #7's own "Total
possible before capping" line), and `max_brake_rate` was already documented
as a supporting-only field. No code change was made for this instruction;
`docs/evaluator.md` #7 carries the same note, and
`tests/test_evaluator.py::test_max_brake_rate_has_no_effect_on_severity_score`
locks in that a swing in it alone doesn't move `severity_score`.

## 2. `delta_severity` — the real search objective

Every search method's actual objective is
`trial.metrics.severity_score - baseline.severity_score` for the scenario
in use, not raw `severity_score` (raw stays for descriptive stats -
"how bad was this trial in absolute terms" - delta is what's actually
being maximized). `Trial` (`avredteam_carla/agents/campaign.py`) carries
both directly: a `baseline_severity` field set by the caller, and a
`delta_severity` property computed from it and `metrics.severity_score`
(`None` if either input is missing, e.g. an `infra_failure` trial with no
metrics at all). `CampaignResult` gained `sorted_by_delta_severity()`/
`best_trial_by_delta()` alongside the existing raw-severity versions.

`baseline_metrics` is fetched once per campaign by the *caller* (not
inside each `SearchMethod.run_campaign()`), so a verification run
comparing all three methods on one scenario only pays for one baseline
episode, not three - see `verify_phase4.py`.

## 3. Failed-trial policy — a real decision, not left open

A trial that exhausts its subprocess retries (Step 5) is logged as
`Trial(outcome="infra_failure", metrics=None, error=...)`, never as a
`Trial` with `severity_score=0` - a real "nothing dangerous happened
because there was no episode" outcome would be indistinguishable from
"the infrastructure died" if both were scored 0, silently polluting every
downstream severity comparison and analysis. `CampaignResult.trials`
includes `infra_failure` entries; `successful_trials()`/
`sorted_by_severity()`/`sorted_by_delta_severity()` all exclude them.

**Does an `infra_failure` trial consume budget, or get replaced?**
Decided: **it consumes budget** (does not get replaced with another
attempt). Simpler, and bounded worst-case runtime - the alternative
(retry until a slot succeeds) is fairer in principle but has unbounded
worst-case runtime on a node `docs/evaluator.md`'s own Phase 3
investigation already found runs bursty (CPU-load spikes correlating with
crashes) and near-full on disk; a single stuck/overloaded window could
otherwise stall an entire campaign indefinitely. All three search methods
(`RandomSearch`, `BayesianSearch`, `LLMAgentSearch`) apply this
consistently - `BayesianSearch` additionally has to give Optuna's
objective a real number for a failed trial, since it can't return `None`;
it uses a fixed `INFRA_FAILURE_OBJECTIVE = -1000.0`, well outside
`delta_severity`'s real possible range of `[-100, 100]`, so TPE learns to
avoid failure-prone regions without ever confusing one for a genuinely
low-severity outcome.

## 4. Shared `SearchMethod` interface (Step 1)

`avredteam_carla/agents/search.py`:

```python
class SearchMethod:
    name: str
    def run_campaign(self, scenario, budget, seed, run_trial, baseline_metrics) -> CampaignResult: ...
```

Every method must, and is unit-tested to: run exactly `budget` trials (a
failed trial still consumes a slot, per §3 above - so `budget` in,
`budget` `Trial`s out, always), be reproducible for a fixed `seed`, and
populate `CampaignResult.agent_notes` with one rationale string per trial
in order (even random search's "trial N: uniform random sample,
attack=... params=..." counts - this keeps later analysis code uniform
across methods rather than treating notes as an LLM-agent-only feature).

`run_trial` is injected as a plain callable
(`avredteam_carla.agents.search.TrialRunner`, matching
`avredteam_carla.runner.run_trial`'s real signature: `(scenario,
attack_name, attack_params, baseline_severity) -> Trial`) rather than
called internally, so every method is unit-tested with a stub that never
touches CARLA. The one production `TrialRunner` Phase 4 ships is
`avredteam_carla.agents.isolated_runner.run_trial_isolated` (§7 below).

`attack_pool(scenario)` and `tunable_params_for(attack_cls)` read
`avredteam_carla.attacks.registry.ATTACK_REGISTRY` programmatically -
every registered attack, or just `scenario.fixed_attack_name` if set
(Step 2's "fixed attack type if scenario config specifies one - support
both", now a real field on `ScenarioConfig`) - so a future 4th attack
type needs no change in any search method.

## 5. Random search (Step 2)

`avredteam_carla/agents/random_search.py` - the floor baseline, uniform
sampling over the candidate pool's `TunableParam` ranges (`sample_uniform_params`
in `search.py`, shared with the LLM agent's fallback path), one attack
type picked uniformly per trial via `random.Random(seed)`. Deliberately no
adaptation, no memory of past trials.

## 6. Bayesian optimization (Step 3)

`avredteam_carla/agents/bayesian_search.py` - **one Optuna TPE study per
campaign**, not one study per attack type (the brief left this open;
decided and documented here rather than guessed silently). `budget` is a
single fixed integer for the whole campaign (§4's exact-budget
requirement); splitting it across N separate per-attack studies up front
would give up exactly the adaptive exploration Bayesian optimization is
meant to add over random search, since there'd be no way to shift trials
toward whichever attack type is proving effective once a study starts. A
single study with `attack_name` sampled as a categorical parameter
alongside each attack's own params is also Optuna's own documented
pattern for a conditional/mixed search space (a "define-by-run" objective
that only calls `suggest_*()` for the branch it actually takes) - not a
workaround, the intended way to express this.

Search space is read programmatically from each attack's
`tunable_params` (never hand-duplicated), namespaced per-attack
(`f"{attack_name}.{param.name}"`) so two attacks' same-named params never
share one TPE distribution even if their ranges happen to differ.
Objective is `delta_severity` (§2); an `infra_failure` trial is scored
per §3.

## 7. LLM agent (Step 4)

`avredteam_carla/agents/llm_agent_search.py` - a Claude tool-use loop.
Reimplemented directly against this repo's real interfaces
(`SearchMethod`/`TunableParam`/`run_trial`) - **not** built on or
importing any standalone-prototype orchestrator, since no such repo
exists anywhere in this project's history (checked, not assumed).

- **Tool**: `run_attack_trial(attack_name, attack_params, reasoning)`.
  `input_schema`'s `attack_name` enum and the system prompt's per-attack
  parameter descriptions are both generated programmatically from
  `tunable_params_for()`, never hand-written prose that could drift out
  of sync with the real attacks.
- **System prompt** states the objective explicitly (maximize
  `delta_severity` within the stated budget, baseline severity given as a
  number) and instructs explore-then-exploit (diverse early trials, then
  refine the best-so-far attack/region).
- **Budget enforcement, both directions**: only `run_attack_trial` is
  offered - there is no "finish early" tool, so under-spending isn't in
  the model's action space at all. Over-spending is prevented by the loop
  itself: once `budget` trials have run, no further model calls happen,
  and any extra `tool_use` blocks already in flight in one turn get a
  `{"skipped": "budget exhausted"}` `tool_result` instead of being
  executed.
- **Model doesn't cooperate**: if it stalls (no `tool_use` in a turn) or
  keeps sending malformed requests (unknown `attack_name`/param key) past
  `MAX_EXTRA_TURNS` (10) extra turns, the loop gives up and fills the
  remainder of the budget with uniform random samples
  (`sample_uniform_params`, the same function random search uses) - the
  exact-budget contract holds unconditionally, never left hoping the
  model behaves.
- Model is configurable (`LLMAgentSearch(model=...)`), defaulting to
  `claude-sonnet-5` - a sensible present-day default, not a guarantee of
  what's actually available/deployed when this runs for real; confirm
  against the account's available models before a real campaign.
- The client is injected (`LLMAgentSearch(client=...)`), so the entire
  loop - budget accounting, multi-tool-use-per-turn handling,
  malformed-request handling, the stall fallback - is unit-tested end to
  end against a scripted fake client
  (`tests/test_llm_agent_search.py`). No real model call is possible from
  this sandbox.

**Update (real-hardware verification): no `anthropic` SDK version can
actually run on this project's Python 3.7 - not a missing wheel, a hard
version floor.** Checked directly rather than assumed: every SDK release
from 0.27.0 onward (the first with tool-use support) depends on `jiter`,
and `jiter`'s package metadata declares `Requires-Python: >=3.8` on every
version ever published, including the first - `pip` refuses to even
attempt building it, before any compile step runs. Installed a real local
Rust toolchain (`rustup`, user-level, no sudo) specifically to rule out
"just no prebuilt wheel for this platform" as the cause; it made no
difference, confirming the declared `Requires-Python` floor is the actual
blocker. The only cp37-installable `anthropic` version, 0.26.0, predates
the `tools` parameter on `messages.create()` entirely (confirmed via
`inspect.signature`). **Decision (this option chosen over a two-environment
split or deferring Step 4's real run - see the PR discussion): the default
client is now `avredteam_carla.agents.anthropic_http_client.AnthropicHTTPClient`**,
a small hand-written client hitting the same
`https://api.anthropic.com/v1/messages` REST endpoint directly via `httpx`
(pure Python, already an `anthropic==0.26.0` transitive dependency, cp37-
compatible), exposing just the `.messages.create(...)` surface
`LLMAgentSearch` uses - no other line of `llm_agent_search.py` needed to
change beyond the default-client factory. Unit-tested against
`httpx.MockTransport` (`tests/test_anthropic_http_client.py`) - request
shape, response-block attribute access, the response-content-fed-back-into-
the-next-request round-trip `LLMAgentSearch`'s loop actually does, and
retry behavior on 429/5xx vs. immediate-raise on other 4xx. Trade-off
accepted knowingly: no SDK-provided auto-retry/error-taxonomy/streaming -
this shim only implements what `LLMAgentSearch` actually calls.

## 8. Infrastructure hardening (Step 5) - applies to all three methods

**Subprocess-per-trial isolation - mandatory, not optional.**
`avredteam_carla/agents/isolated_runner.py`'s `run_trial_isolated()` is
the one production `TrialRunner`: every trial re-invokes
`avredteam_carla.agents.trial_worker` as a fresh subprocess, the same
self-reinvocation shape `verify_phase3.py` already proved necessary -
Phase 3's real finding (`docs/evaluator.md` #8) was that repeated
in-process `run_trial()` calls can abort the whole process via an
uncaught C++ exception (`terminate called after throwing
carla::client::TimeoutException`) that no Python `try`/`except` can
catch. A process exit trivially and completely tears down every
socket/thread/GPU context; nothing less does.

**Retry-with-backoff**, bounded at 3 attempts
(`isolated_runner.DEFAULT_RETRIES`, matching `verify_phase3.py`'s proven
`STAGE_SUBPROCESS_RETRIES` - same node, same cap, no reason to pick a
different number without new evidence), exponential backoff starting at
8s (`DEFAULT_BASE_BACKOFF_S`, matching `verify_phase3.py`'s proven
`SETTLE_SLEEP_S`) doubling up to a 60s cap. Justified directly by
`docs/evaluator.md`'s "Disk vs. CPU load, isolated" finding: this node's
load is bursty, not steadily high, and a quiet window reliably followed
within minutes in every observation there - a short, growing wait is
expected to ride out a spike rather than needing an unbounded one.
`subprocess_timeout_s` defaults to 1800s (30 min) - generous against a
truly hung subprocess (a real episode ran 600-700s in Phase 3), not a
per-RPC-call timeout (CARLA's own client-side `load_world()` timeout,
60s, is unrelated and untouched here).

**Pre-flight check, once per campaign, not once per trial.**
`avredteam_carla/preflight.py`'s `preflight_snapshot()`: disk headroom
(`shutil.disk_usage`), per-GPU utilization/memory (`nvidia-smi`, parsed
by `parse_nvidia_smi_csv`), load average (`os.getloadavg`), and an
automatic idle-GPU pick (`pick_idle_gpu` - lowest utilization, ties broken
by lowest memory used) extending TASK.md's original Phase 1 decision
("prefer an idle GPU... never hardcoded") from a human running `nvidia-smi`
by hand to the campaign runner doing it once per campaign. Logged into
`CampaignResult.preflight` so later analysis can check whether a
campaign's anomalies correlated with node conditions at start time -
directly motivated by Phase 3's finding that crashes tracked real-time
load/disk pressure, not a fixed call count or GPU choice.

**Failed-trial policy**: §3 above.

**Visual sanity check, every attacked trial.**
`avredteam_carla/attacks/sanity_frames.py`'s `SanityFrameTracker` captures
3 representative BEV frame pairs per trial (start, midpoint, worst-moment)
instead of a full-episode dump - see the module's own docstring for the
online doubling heuristic (midpoint) and per-tick proxy score
(worst-moment) used to do this without knowing the final episode length
or the true whole-episode `severity_score` in advance, and without ever
buffering more than one candidate frame pair per slot. Wired into
`run_episode()` via a new `sanity_frames_dir` parameter (only active
alongside an attack; separate from Phase 2's still-unchanged periodic
`--bev-frames-every`), threaded through `runner.run_trial()` and
`trial_worker.py`.

**Every output path under `/data/$USER`, confirmed rather than assumed.**
`verify_phase4.py` checks `--out` and `--sanity-frames-root` actually
resolve under `/data/` before running anything (`_require_under_data_dir`)
- this repo checkout itself lives under `/home` (quota-limited on Maui
per TASK.md's original decision), so a relative default here would
silently violate that. `run_clean_episode.py --sanity-frames-dir` carries
the same instruction in its `--help` text (no code-level guard there,
since Phase 1-3's existing `--out`/`--bev-frames-dir` flags were never
guarded either - kept consistent with that established CLI convention
rather than introducing a one-off exception).

## 9. Step 6 verification - needs real Maui hardware

`avredteam_carla/verify_phase4.py` runs all three methods against Town01
with a small shared budget (default 10) and a fixed seed, through the
real Step 5 infrastructure, and prints:

| Method | Trials completed | Infra failures | Best delta_severity | Wall-clock time |
|---|---|---|---|---|
| Random search | | | | |
| Bayesian opt | | | | |
| LLM agent | | | | |

**Not runnable in this dev sandbox** - needs a live CARLA server,
`pip install optuna anthropic` in the `carla-redteam` env, and
`ANTHROPIC_API_KEY` set. CLI parsing and the `/data` path guard are
confirmed working standalone (no CARLA needed for either). A real-hardware
session should run:

```bash
source /data/savyo/carla-redteam/env.sh
cd ~/av-redteam-agent && git checkout phase-4-search-methods
pip install optuna anthropic
export ANTHROPIC_API_KEY=...   # not yet in env.sh - add it there once confirmed working

python -m pytest tests/ -q   # should be 137 passed, same as the dev sandbox

# CARLA server already running per docs/setup.md
python -m avredteam_carla.verify_phase4 \
  --roach-root "$PROJECT_DATA_DIR/roach" --host localhost --port 2100 \
  --budget 10 --search-seed 2021 \
  --sanity-frames-root "$PROJECT_DATA_DIR/phase4_verification/frames" \
  --out "$PROJECT_DATA_DIR/phase4_verification/results.json"
```

and paste the printed acceptance table (plus at least one observed
retry-recovery, if the node is under any load during the run) back into
this section and the PR description.

## Explicitly out of scope for this phase

No scenario suite beyond Town01/route-0 (Phase 5). No full budget x seed x
scenario sweep at scale (Phase 6) - this phase proves the three methods
work end-to-end, Phase 6 is where they run at the scale the paper needs.

## Dependencies not yet in any requirements file

This repo has no `requirements.txt`/`setup.py` (dependencies have so far
been listed inline in each phase's docs, e.g. `docs/evaluator.md` #8's
`pip install pytest`, kept consistent here rather than introducing a new
convention). Phase 4 adds two: `optuna` (Bayesian search) and `anthropic`
(LLM agent). Both were installed and exercised against real code paths in
this dev sandbox (Optuna's real TPE sampler; a scripted fake standing in
only for the Anthropic *client*, never for the SDK's real
request/response object shapes) - `pip install optuna anthropic` in the
`carla-redteam` env before running `verify_phase4.py` for real.
