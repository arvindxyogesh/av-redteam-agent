"""Unit tests for LLMAgentSearch (docs/search_methods.md Step 4). The
Anthropic client is stubbed with a scripted fake exposing the same
`.messages.create(...)` surface used by the real SDK - no network, no API
key needed. See module docstring in llm_agent_search.py for the loop's
budget/stall/malformed-request contract being tested here."""
from types import SimpleNamespace

import pytest

from avredteam_carla.agents.campaign import Trial, TRIAL_OUTCOME_INFRA_FAILURE
from avredteam_carla.agents.llm_agent_search import LLMAgentSearch, MAX_EXTRA_TURNS, TOOL_NAME
from avredteam_carla.evaluator import evaluate
from avredteam_carla.runner import ScenarioConfig
from tests.test_evaluator import make_log

BASELINE_METRICS = evaluate(make_log(steer=[0.0] * 5, termination_reason="route_completed"))


def _scenario(**kwargs):
    return ScenarioConfig(name="s", roach_root="/fake", **kwargs)


def _stub_run_trial(calls):
    def run_trial(scenario, attack_name, attack_params, baseline_severity):
        calls.append((attack_name, dict(attack_params)))
        m = evaluate(make_log(steer=[0.0] * 5, termination_reason="route_completed"))
        return Trial(scenario.name, attack_name, attack_params, m, baseline_severity=baseline_severity)

    return run_trial


def tool_use(id_, attack_name, attack_params=None, reasoning="test"):
    return SimpleNamespace(
        type="tool_use", id=id_,
        input={"attack_name": attack_name, "attack_params": attack_params or {}, "reasoning": reasoning},
    )


def text(msg="thinking"):
    return SimpleNamespace(type="text", text=msg)


class FakeMessages:
    def __init__(self, scripted_responses):
        self._responses = list(scripted_responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        assert self._responses, "FakeClient ran out of scripted responses"
        content = self._responses.pop(0)
        return SimpleNamespace(content=content, stop_reason="tool_use")


class FakeClient:
    def __init__(self, scripted_responses):
        self.messages = FakeMessages(scripted_responses)


def _agent(scripted_responses):
    return LLMAgentSearch(client=FakeClient(scripted_responses))


# --- happy path ------------------------------------------------------


def test_llm_agent_runs_exactly_budget_trials_one_tool_use_per_turn():
    calls = []
    scripted = [[tool_use(f"id{i}", "channel_noise", {"amplitude": 50.0})] for i in range(4)]
    campaign = _agent(scripted).run_campaign(
        _scenario(), budget=4, seed=0, run_trial=_stub_run_trial(calls), baseline_metrics=BASELINE_METRICS
    )
    assert len(calls) == 4
    assert len(campaign.trials) == 4
    assert len(campaign.agent_notes) == 4
    assert all("LLM agent" in n for n in campaign.agent_notes)


def test_llm_agent_handles_multiple_tool_uses_in_one_turn():
    calls = []
    scripted = [[tool_use("a", "channel_noise"), tool_use("b", "phantom_actor")]]
    campaign = _agent(scripted).run_campaign(
        _scenario(), budget=2, seed=0, run_trial=_stub_run_trial(calls), baseline_metrics=BASELINE_METRICS
    )
    assert len(calls) == 2
    assert len(campaign.trials) == 2


def test_llm_agent_stops_running_trials_once_budget_reached_mid_turn():
    """3 tool_use blocks in one turn, budget=2 - the 3rd must be skipped,
    not executed (Step 4's "must not over-spend budget")."""
    calls = []
    scripted = [[tool_use("a", "channel_noise"), tool_use("b", "phantom_actor"), tool_use("c", "geometry_spoof")]]
    campaign = _agent(scripted).run_campaign(
        _scenario(), budget=2, seed=0, run_trial=_stub_run_trial(calls), baseline_metrics=BASELINE_METRICS
    )
    assert len(calls) == 2
    assert len(campaign.trials) == 2


def test_llm_agent_system_prompt_states_objective_and_budget():
    calls = []
    scripted = [[tool_use("a", "channel_noise")] for _ in range(2)]
    client = FakeClient(scripted)
    LLMAgentSearch(client=client).run_campaign(
        _scenario(), budget=2, seed=0, run_trial=_stub_run_trial(calls), baseline_metrics=BASELINE_METRICS
    )
    system_prompt = client.messages.calls[0]["system"]
    assert "delta_severity" in system_prompt
    assert "2" in system_prompt  # budget mentioned
    assert "channel_noise" in system_prompt and "phantom_actor" in system_prompt


def test_llm_agent_tool_schema_lists_registered_attacks():
    calls = []
    scripted = [[tool_use("a", "channel_noise")] for _ in range(1)]
    client = FakeClient(scripted)
    LLMAgentSearch(client=client).run_campaign(
        _scenario(), budget=1, seed=0, run_trial=_stub_run_trial(calls), baseline_metrics=BASELINE_METRICS
    )
    tool = client.messages.calls[0]["tools"][0]
    assert tool["name"] == TOOL_NAME
    assert set(tool["input_schema"]["properties"]["attack_name"]["enum"]) >= {"channel_noise", "geometry_spoof", "phantom_actor"}


# --- malformed requests --------------------------------------------------


def test_llm_agent_malformed_attack_name_does_not_consume_budget():
    calls = []
    scripted = [
        [tool_use("bad", "not_a_real_attack")],
        [tool_use("good", "channel_noise")],
    ]
    campaign = _agent(scripted).run_campaign(
        _scenario(), budget=1, seed=0, run_trial=_stub_run_trial(calls), baseline_metrics=BASELINE_METRICS
    )
    assert len(calls) == 1  # only the valid one actually ran an episode
    assert len(campaign.trials) == 1


def test_llm_agent_malformed_param_name_does_not_consume_budget():
    calls = []
    scripted = [
        [tool_use("bad", "channel_noise", {"not_a_real_param": 1.0})],
        [tool_use("good", "channel_noise", {"amplitude": 50.0})],
    ]
    campaign = _agent(scripted).run_campaign(
        _scenario(), budget=1, seed=0, run_trial=_stub_run_trial(calls), baseline_metrics=BASELINE_METRICS
    )
    assert len(calls) == 1


# --- infra_failure passthrough -------------------------------------------


def test_llm_agent_reports_infra_failure_to_model_and_still_consumes_budget():
    def flaky_run_trial(scenario, attack_name, attack_params, baseline_severity):
        return Trial(scenario.name, attack_name, attack_params, None, outcome=TRIAL_OUTCOME_INFRA_FAILURE, error="timeout")

    scripted = [[tool_use("a", "channel_noise")]]
    campaign = _agent(scripted).run_campaign(
        _scenario(), budget=1, seed=0, run_trial=flaky_run_trial, baseline_metrics=BASELINE_METRICS
    )
    assert len(campaign.trials) == 1
    assert campaign.trials[0].outcome == TRIAL_OUTCOME_INFRA_FAILURE


# --- stall fallback --------------------------------------------------------


def test_llm_agent_falls_back_to_random_after_persistent_stall():
    calls = []
    budget = 2
    # Every turn returns text only (no tool_use) - the model never
    # cooperates. After budget + MAX_EXTRA_TURNS such turns, the loop must
    # give up and fill the remainder with random samples rather than loop
    # forever or under-spend the budget.
    scripted = [[text()] for _ in range(budget + MAX_EXTRA_TURNS)]
    campaign = _agent(scripted).run_campaign(
        _scenario(), budget=budget, seed=0, run_trial=_stub_run_trial(calls), baseline_metrics=BASELINE_METRICS
    )
    assert len(campaign.trials) == budget
    assert all("fallback" in n for n in campaign.agent_notes)


def test_llm_agent_fixed_attack_name_respected_in_tool_schema_and_fallback():
    calls = []
    budget = 1
    scripted = [[text()] for _ in range(budget + MAX_EXTRA_TURNS)]
    client = FakeClient(scripted)
    campaign = LLMAgentSearch(client=client).run_campaign(
        _scenario(fixed_attack_name="phantom_actor"), budget=budget, seed=0,
        run_trial=_stub_run_trial(calls), baseline_metrics=BASELINE_METRICS,
    )
    assert client.messages.calls[0]["tools"][0]["input_schema"]["properties"]["attack_name"]["enum"] == ["phantom_actor"]
    assert calls == [("phantom_actor", {})] or calls[0][0] == "phantom_actor"
