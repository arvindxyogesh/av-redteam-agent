# av-redteam-agent

Research project: can an LLM agent discover sensor-level adversarial attacks
on a learning-based autonomous driving planner (Roach, running in CARLA)
more efficiently than classical black-box search? This repo currently holds
**Phase 1** (CARLA + Roach infrastructure), **Phase 2** (a BEV-space attack
library), and **Phase 3** (formal metrics + `Trial`/`CampaignResult`
plumbing) — no search methods or LLM agent integration yet, that's Phase 4.

- Full task briefs and the decisions made along the way: [`TASK.md`](TASK.md)
- Phase 1 install + run instructions: [`docs/setup.md`](docs/setup.md)
- Phase 2 observation-space docs + attack interface: [`docs/attacks.md`](docs/attacks.md)
- Phase 3 metric definitions + severity weighting: [`docs/evaluator.md`](docs/evaluator.md)
- `scripts/launch_carla.sh` — launch a headless CARLA server (GPU + port parameterized)
- `scripts/check_client_connection.py` — verify a CARLA client connection
- `avredteam_carla/run_clean_episode.py` — run one Roach episode (clean, or with `--attack`) and log it; `run_episode()` is the importable core, `main()` a thin CLI wrapper
- `avredteam_carla/attacks/` — the BEV-space attack library (`channel_noise`, `geometry_spoof`, `phantom_actor`) and the `RlBirdviewWrapper.process_obs` interception hook
- `avredteam_carla/ground_truth.py` — real-CARLA-state signals (route deviation, obstacle clearance) the evaluator needs but `carla_gym` doesn't expose cleanly
- `avredteam_carla/evaluator.py` — `EpisodeMetrics` + `evaluate(log)`, per `docs/evaluator.md`
- `avredteam_carla/agents/campaign.py` — `Trial`/`CampaignResult`, the shared interface Phase 4's search methods will use
- `avredteam_carla/runner.py` — `run_trial()`/`run_baseline()`, what a search method actually calls
- `avredteam_carla/verify_phase3.py` — runs baseline + all three attacks + a repeated-call stability check, prints the acceptance table
- `avredteam_carla/compare_episodes.py` — deviation metrics between a clean and an attacked episode log
- `tests/` — unit tests (pure Python/numpy, no CARLA needed: `pytest tests/`)
