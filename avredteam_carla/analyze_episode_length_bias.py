"""Phase 4 Step 0: does any EpisodeMetrics field carry a systematic bias
correlated with episode length/tick-count, independent of attack severity?
(docs/search_methods.md Step 0 - required to run and resolve *before*
anything else in Phase 4, since a wrong call here propagates through every
later phase.)

What this module can and can't establish without a live CARLA server:

**Establishable now, from the evaluator's own code, no real logs needed**
(see analyze_prefix_length_sensitivity() and its unit tests in
tests/test_analyze_episode_length_bias.py): `max_lateral_offset` and
`min_obstacle_clearance` are a running max and a running min, taken over
the *entire* episode's tick series (docs/evaluator.md #3/#6). That makes
their length-sensitivity a mathematical certainty, not something that
needs real data to discover: a min taken over a longer prefix of the same
series can only stay the same or decrease as more ticks are added (a min
over a superset is <= a min over any subset), and symmetrically a max can
only stay the same or increase. So, holding the *true* underlying attack
severity fixed, a longer episode's `min_obstacle_clearance` is biased
downward and its `max_lateral_offset` is biased upward, purely from having
more ticks/opportunities - and `min_obstacle_clearance` feeds directly
into `severity_score`'s `(2 - clearance) * 5` term (docs/evaluator.md #7),
so this is a real, structural length-sensitivity in the composite score
itself, not just a curiosity about a supporting field.

By contrast, `off_lane_frac` and `chattering_rate` are already
rate/fraction-normalized (a count divided by `n_ticks` or by the number of
consecutive-tick pairs compared - docs/evaluator.md #1/#3), so they don't
carry this same structural bias: adding more ticks to a stationary-ish
process changes a fraction's *precision* (small-sample noise shrinks), not
its *expected direction*.

**NOT establishable without real per-tick logs**: how large this bias
actually is on Phase 3's real runs. Phase 3's four real logs
(baseline/channel_noise/geometry_spoof/phantom_actor) spanned genuinely
different tick counts (1355-3104, per docs/evaluator.md's acceptance
table) *because* they represent different attacks - so a raw
length-vs-metric correlation across just those four points can't separate
"this attack caused both a different episode length and a different
severity" from "episode length alone moved the metric." Isolating length
as the only varying factor needs either same-condition-different-length
real runs, or (cheaper, and what this module supports) truncating one
real log's own tick series into shorter prefixes and re-running evaluate()
on each prefix - see analyze_prefix_length_sensitivity() below. Neither
requires a live CARLA server *once a real log file exists*, but this dev
sandbox has none: Phase 3's logs are gitignored (see .gitignore's
`/logs/*`) and live only on the Maui cluster's filesystem, never checked
into this repo. See docs/search_methods.md's Step 0 section for the
concrete real-hardware follow-up this leaves open.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from avredteam_carla.evaluator import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("analyze_episode_length_bias")

# The EpisodeMetrics fields worth checking: every numeric field that feeds
# severity_score, plus the two ground-truth extrema fields (docs/evaluator.md
# #3/#6) flagged above as structurally length-sensitive by construction.
CANDIDATE_METRICS = (
    "chattering_rate",
    "mean_abs_steering_rate",
    "max_lateral_offset",
    "off_lane_frac",
    "min_obstacle_clearance",
    "severity_score",
)


def pearson_r(xs: List[float], ys: List[float]) -> Optional[float]:
    """Plain-stdlib Pearson correlation coefficient - no new numeric
    dependency for a handful of data points. None if undefined (fewer
    than 2 points, or zero variance in either series)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x * var_y) ** 0.5


def truncate_log(log_dict: dict, n_ticks: int) -> dict:
    """A new log dict whose "ticks" is the first n_ticks entries of the
    original - everything else about the trajectory up to that point is
    untouched (same real driving, just cut short), isolating episode
    length as the one varying factor. meta.termination_reason is
    deliberately left as the original's (a truncated log didn't actually
    terminate at n_ticks - evaluate() doesn't read termination_reason for
    anything this module checks, only "collided"/"completed", and neither
    is under test here)."""
    truncated = copy.deepcopy(log_dict)
    truncated["ticks"] = truncated["ticks"][:n_ticks]
    return truncated


def analyze_prefix_length_sensitivity(log_dict: dict, fractions=(0.1, 0.25, 0.5, 0.75, 1.0)) -> dict:
    """Evaluates increasingly long prefixes of one real (or synthetic) log
    and reports each CANDIDATE_METRICS field's value at each prefix length,
    plus its Pearson correlation with n_ticks across those prefixes. This
    is the concrete Step 0 check run against one real Phase 3 log by a
    real-hardware follow-up session (see module docstring) - or against
    synthetic data here, where it's unit-tested (tests/
    test_analyze_episode_length_bias.py) to lock in the min/max monotonicity
    argument above as a property of evaluate() itself, not an assumption.
    """
    n_ticks_total = len(log_dict["ticks"])
    lengths = sorted({max(1, round(n_ticks_total * f)) for f in fractions if f <= 1.0} | {n_ticks_total})

    rows = []
    for n in lengths:
        m = evaluate(truncate_log(log_dict, n))
        rows.append({"n_ticks": n, **{field: getattr(m, field) for field in CANDIDATE_METRICS}})

    correlations = {}
    for field in CANDIDATE_METRICS:
        xs = [r["n_ticks"] for r in rows]
        ys = [r[field] for r in rows if r[field] is not None]
        if len(ys) == len(rows):
            correlations[field] = pearson_r(xs, ys)
        else:
            correlations[field] = None  # field not populated in this log (e.g. no ground-truth fields)

    return {"n_ticks_total": n_ticks_total, "rows": rows, "correlation_with_n_ticks": correlations}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--log", action="append", dest="logs", default=[], required=True,
        help="Path to a real episode log JSON (run_clean_episode.py's --out format). Repeatable - "
        "pass Phase 3's baseline + all three attack logs to reproduce Step 0's cross-condition check.",
    )
    p.add_argument("--out", required=True, help="Where to write the full JSON report")
    args = p.parse_args()

    report = {}
    for log_path in args.logs:
        log_dict = json.loads(Path(log_path).read_text())
        name = log_dict.get("meta", {}).get("attack", {})
        name = name["name"] if name else "baseline"
        log.info("Analyzing %s (%s, %d ticks)...", log_path, name, len(log_dict["ticks"]))
        report[name] = analyze_prefix_length_sensitivity(log_dict)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("Wrote length-sensitivity report to %s", out_path)

    print()
    print("Prefix-length sensitivity (Pearson r vs. n_ticks, within each condition's own log):")
    print("| Condition | n_ticks (full) | " + " | ".join(CANDIDATE_METRICS) + " |")
    print("|---|---|" + "---|" * len(CANDIDATE_METRICS))
    for name, result in report.items():
        corrs = result["correlation_with_n_ticks"]
        cells = " | ".join(f"{corrs[f]:.2f}" if corrs[f] is not None else "n/a" for f in CANDIDATE_METRICS)
        print(f"| {name} | {result['n_ticks_total']} | {cells} |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
