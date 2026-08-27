# av-redteam-agent

Research project: can an LLM agent discover sensor-level adversarial attacks
on a learning-based autonomous driving planner (Roach, running in CARLA)
more efficiently than classical black-box search? This repo currently holds
**Phase 1** (CARLA + Roach infrastructure) and **Phase 2** (a BEV-space
attack library) — no search methods or LLM agent integration yet, that's
Phase 4.

- Full task briefs and the decisions made along the way: [`TASK.md`](TASK.md)
- Phase 1 install + run instructions: [`docs/setup.md`](docs/setup.md)
- Phase 2 observation-space docs + attack interface: [`docs/attacks.md`](docs/attacks.md)
- `scripts/launch_carla.sh` — launch a headless CARLA server (GPU + port parameterized)
- `scripts/check_client_connection.py` — verify a CARLA client connection
- `avredteam_carla/run_clean_episode.py` — run one Roach episode (clean, or with `--attack`) and log it
- `avredteam_carla/attacks/` — the BEV-space attack library (`channel_noise`, `geometry_spoof`, `phantom_actor`) and the `RlBirdviewWrapper.process_obs` interception hook
- `avredteam_carla/compare_episodes.py` — deviation metrics between a clean and an attacked episode log
- `tests/` — unit tests for the attack library (pure numpy, no CARLA needed: `pytest tests/`)
