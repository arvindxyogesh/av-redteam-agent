"""Unit tests for avredteam_carla.preflight (docs/search_methods.md
Step 5's pre-flight check). subprocess/nvidia-smi is stubbed via
gpu_snapshot()'s injectable runner - no real GPU needed to test the
parsing/picking logic; disk_snapshot()/load_average_snapshot() exercise
the real stdlib calls since they work in any sandbox."""
import subprocess

import pytest

from avredteam_carla.preflight import (
    disk_snapshot,
    gpu_snapshot,
    load_average_snapshot,
    parse_nvidia_smi_csv,
    pick_idle_gpu,
    preflight_snapshot,
)

SAMPLE_CSV = "0, 92, 40000, 40960\n1, 0, 0, 40960\n2, 15, 500, 40960\n"


def test_parse_nvidia_smi_csv():
    gpus = parse_nvidia_smi_csv(SAMPLE_CSV)
    assert gpus == [
        {"index": 0, "utilization_pct": 92.0, "memory_used_mb": 40000.0, "memory_total_mb": 40960.0},
        {"index": 1, "utilization_pct": 0.0, "memory_used_mb": 0.0, "memory_total_mb": 40960.0},
        {"index": 2, "utilization_pct": 15.0, "memory_used_mb": 500.0, "memory_total_mb": 40960.0},
    ]


def test_parse_nvidia_smi_csv_ignores_blank_lines():
    gpus = parse_nvidia_smi_csv("0, 0, 0, 40960\n\n1, 0, 0, 40960\n")
    assert len(gpus) == 2


def test_pick_idle_gpu_lowest_utilization():
    gpus = parse_nvidia_smi_csv(SAMPLE_CSV)
    assert pick_idle_gpu(gpus) == 1


def test_pick_idle_gpu_respects_exclude():
    gpus = parse_nvidia_smi_csv(SAMPLE_CSV)
    assert pick_idle_gpu(gpus, exclude=[1]) == 2


def test_pick_idle_gpu_empty_list_returns_none():
    assert pick_idle_gpu([]) is None


def test_pick_idle_gpu_all_excluded_returns_none():
    gpus = parse_nvidia_smi_csv(SAMPLE_CSV)
    assert pick_idle_gpu(gpus, exclude=[0, 1, 2]) is None


def test_pick_idle_gpu_ties_broken_by_memory():
    gpus = [
        {"index": 5, "utilization_pct": 0.0, "memory_used_mb": 1000.0, "memory_total_mb": 40960.0},
        {"index": 6, "utilization_pct": 0.0, "memory_used_mb": 200.0, "memory_total_mb": 40960.0},
    ]
    assert pick_idle_gpu(gpus) == 6


def test_gpu_snapshot_returns_none_when_nvidia_smi_missing():
    def missing_runner(cmd):
        raise FileNotFoundError("nvidia-smi not found")

    assert gpu_snapshot(runner=missing_runner) is None


def test_gpu_snapshot_returns_none_on_subprocess_error():
    def failing_runner(cmd):
        raise subprocess.CalledProcessError(1, cmd)

    assert gpu_snapshot(runner=failing_runner) is None


def test_gpu_snapshot_parses_stub_output():
    def stub_runner(cmd):
        assert "nvidia-smi" in cmd
        return SAMPLE_CSV

    gpus = gpu_snapshot(runner=stub_runner)
    assert len(gpus) == 3


def test_disk_snapshot_has_expected_fields():
    snap = disk_snapshot("/")
    assert snap["total_gb"] > 0
    assert 0.0 <= snap["used_pct"] <= 100.0


def test_load_average_snapshot_on_linux():
    snap = load_average_snapshot()
    # Linux/macOS sandboxes both support getloadavg(); if it's ever None
    # (unsupported platform), the fields must not be asserted below.
    if snap is not None:
        assert snap["load_1m"] >= 0.0


def test_preflight_snapshot_picks_gpu_from_stubbed_runner():
    def stub_runner(cmd):
        return SAMPLE_CSV

    snap = preflight_snapshot(gpu_runner=stub_runner)
    assert snap["picked_gpu"] == 1
    assert "disk" in snap and "load_average" in snap


def test_preflight_snapshot_picked_gpu_none_without_gpus():
    def missing_runner(cmd):
        raise FileNotFoundError()

    snap = preflight_snapshot(gpu_runner=missing_runner)
    assert snap["picked_gpu"] is None
    assert snap["gpus"] is None
