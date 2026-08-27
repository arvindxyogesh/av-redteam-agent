# av-redteam-agent

Research project: can an LLM agent discover sensor-level adversarial attacks
on a learning-based autonomous driving planner (Roach, running in CARLA)
more efficiently than classical black-box search? This repo currently holds
**Phase 1 only**: CARLA + Roach infrastructure, no attacks/search/agent yet.

- Full task brief and the decisions made along the way: [`TASK.md`](TASK.md)
- Reproducible install + run instructions: [`docs/setup.md`](docs/setup.md)
- `scripts/launch_carla.sh` — launch a headless CARLA server (GPU + port parameterized)
- `scripts/check_client_connection.py` — verify a CARLA client connection
- `avredteam_carla/run_clean_episode.py` — run one clean Roach episode and log it
