# av-redteam-agent

Research project: can an LLM agent discover sensor-level adversarial attacks
on a learning-based autonomous driving planner (Roach, running in CARLA)
more efficiently than classical black-box search? This repo currently holds
**Phase 1** (CARLA + Roach infrastructure), **Phase 2** (a BEV-space attack
library), **Phase 3** (formal metrics + `Trial`/`CampaignResult` plumbing),
and **Phase 4** (random search, Bayesian optimization, and an LLM agent, all
sharing one `SearchMethod` interface) — no scenario suite beyond Town01/
route-0 yet, that's Phase 5.

- Full task briefs and the decisions made along the way: [`TASK.md`](TASK.md)
- Phase 1 install + run instructions: [`docs/setup.md`](docs/setup.md)
- Phase 2 observation-space docs + attack interface: [`docs/attacks.md`](docs/attacks.md)
- Phase 3 metric definitions + severity weighting: [`docs/evaluator.md`](docs/evaluator.md)
- Phase 4 search methods, infra hardening, and decisions log: [`docs/search_methods.md`](docs/search_methods.md)
- `scripts/launch_carla.sh` — launch a headless CARLA server (GPU + port parameterized)
- `scripts/check_client_connection.py` — verify a CARLA client connection
- `avredteam_carla/run_clean_episode.py` — run one Roach episode (clean, or with `--attack`) and log it; `run_episode()` is the importable core, `main()` a thin CLI wrapper
- `avredteam_carla/attacks/` — the BEV-space attack library (`channel_noise`, `geometry_spoof`, `phantom_actor`), the `RlBirdviewWrapper.process_obs` interception hook, and `sanity_frames.py` (Phase 4's per-trial visual sanity check)
- `avredteam_carla/ground_truth.py` — real-CARLA-state signals (route deviation, obstacle clearance) the evaluator needs but `carla_gym` doesn't expose cleanly
- `avredteam_carla/evaluator.py` — `EpisodeMetrics` + `evaluate(log)`, per `docs/evaluator.md`
- `avredteam_carla/agents/campaign.py` — `Trial`/`CampaignResult` (`delta_severity`, `TrialOutcome`), the shared interface every Phase 4 search method uses
- `avredteam_carla/agents/search.py` — the shared `SearchMethod` interface + `attack_pool()`/`sample_uniform_params()` helpers
- `avredteam_carla/agents/random_search.py`, `bayesian_search.py`, `llm_agent_search.py` — the three Phase 4 search methods
- `avredteam_carla/agents/isolated_runner.py` + `trial_worker.py` — subprocess-per-trial isolation + retry-with-backoff, the production `TrialRunner` all three methods use
- `avredteam_carla/preflight.py` — disk/GPU/load snapshot + automatic idle-GPU pick, run once per campaign
- `avredteam_carla/analyze_episode_length_bias.py` — Phase 4 Step 0's episode-length-vs-metric bias analysis tool
- `avredteam_carla/runner.py` — `run_trial()`/`run_baseline()`, what a search method actually calls
- `avredteam_carla/verify_phase3.py` / `verify_phase4.py` — real-hardware verification harnesses, print the acceptance table for each phase
- `avredteam_carla/compare_episodes.py` — deviation metrics between a clean and an attacked episode log
- `tests/` — unit tests (pure Python/numpy/optuna, no CARLA needed: `pytest tests/`)
