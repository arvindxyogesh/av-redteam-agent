import pytest

from avredteam_carla.agents.campaign import Trial, CampaignResult
from avredteam_carla.evaluator import evaluate
from tests.test_evaluator import make_log


def metrics(collision_at=None, termination_reason="route_completed"):
    return evaluate(make_log(steer=[0.0] * 5, collision_at=collision_at, termination_reason=termination_reason))


def test_trial_to_dict_contains_all_fields():
    t = Trial(scenario_name="town01_route0", attack_name="phantom_actor", attack_params={"distance_m": 15.0}, metrics=metrics())
    d = t.to_dict()
    assert d["scenario_name"] == "town01_route0"
    assert d["attack_name"] == "phantom_actor"
    assert d["attack_params"] == {"distance_m": 15.0}
    assert "severity_score" in d["metrics"]


def test_campaign_result_add_trial_rejects_scenario_mismatch():
    campaign = CampaignResult(scenario_name="town01_route0", baseline_metrics=metrics())
    bad_trial = Trial(scenario_name="town02_route1", attack_name="channel_noise", attack_params={}, metrics=metrics())
    with pytest.raises(ValueError):
        campaign.add_trial(bad_trial)


def test_sorted_by_severity_descending():
    campaign = CampaignResult(scenario_name="s", baseline_metrics=metrics())
    low = Trial("s", "channel_noise", {}, metrics(termination_reason="route_completed"))
    high = Trial("s", "geometry_spoof", {}, metrics(collision_at=1, termination_reason="collision"))
    campaign.add_trial(low)
    campaign.add_trial(high)
    ranked = campaign.sorted_by_severity()
    assert ranked[0] is high
    assert ranked[1] is low


def test_best_trial_returns_highest_severity():
    campaign = CampaignResult(scenario_name="s", baseline_metrics=metrics())
    low = Trial("s", "a", {}, metrics(termination_reason="route_completed"))
    high = Trial("s", "b", {}, metrics(collision_at=0, termination_reason="collision"))
    campaign.add_trial(low)
    campaign.add_trial(high)
    assert campaign.best_trial() is high


def test_best_trial_none_when_no_trials():
    campaign = CampaignResult(scenario_name="s", baseline_metrics=metrics())
    assert campaign.best_trial() is None


def test_campaign_result_to_dict_round_trips():
    campaign = CampaignResult(scenario_name="s", baseline_metrics=metrics(), agent_notes=["note1"])
    campaign.add_trial(Trial("s", "channel_noise", {"amplitude": 100.0}, metrics(collision_at=2, termination_reason="collision")))
    d = campaign.to_dict()
    assert d["scenario_name"] == "s"
    assert d["agent_notes"] == ["note1"]
    assert len(d["trials"]) == 1
    assert d["trials"][0]["attack_name"] == "channel_noise"
