"""Unit tests for the Phase 2 attack library, against synthetic BEV tensors
matching the exact shapes documented in docs/attacks.md - no CARLA/Roach
needed, so these run anywhere (including CI, and this dev sandbox).

What this DOES verify: shape/dtype preservation, non-mutation of inputs,
parameter validation/clamping, and each attack's core spatial/temporal
behavior on synthetic data.

What this does NOT verify (needs a real Maui run, see docs/attacks.md #5 and
the Phase 2 PR's acceptance table): that these attacks actually change
Roach's control output, or that the hook.py monkeypatch takes effect against
the real RlBirdviewAgent.
"""
import numpy as np
import pytest

from avredteam_carla.attacks import (
    DEFAULT_LAYOUT,
    ChannelNoiseAttack,
    GeometrySpoofAttack,
    PhantomActorAttack,
    build_attack,
)
from avredteam_carla.attacks.base import Attack, TunableParam
from avredteam_carla.attacks._util import shift_columns, draw_disk, squeeze_batch, restore_batch


def make_bev(batch=False):
    rng = np.random.default_rng(0)
    bev = (rng.random((DEFAULT_LAYOUT.num_channels, DEFAULT_LAYOUT.width_px, DEFAULT_LAYOUT.width_px)) > 0.9)
    bev = (bev * 255).astype(np.uint8)
    state = np.zeros(6, dtype=np.float32)
    if batch:
        return bev[np.newaxis], state[np.newaxis]
    return bev, state


# ---------------------------------------------------------------------------
# base.py: TunableParam + Attack plumbing
# ---------------------------------------------------------------------------

def test_tunable_param_clips_out_of_range_values():
    p = TunableParam("x", "float", default=1.0, low=0.0, high=10.0)
    assert p.cast(15.0) == 10.0
    assert p.cast(-5.0) == 0.0
    assert p.cast(3.0) == 3.0


def test_attack_rejects_unknown_param():
    with pytest.raises(ValueError):
        ChannelNoiseAttack(not_a_real_param=1)


def test_attack_uses_defaults_when_no_overrides():
    a = ChannelNoiseAttack()
    assert a.params["channel"] == DEFAULT_LAYOUT.route
    assert a.params["amplitude"] == 80.0


def test_attack_base_apply_is_abstract():
    with pytest.raises(NotImplementedError):
        Attack().apply(*make_bev(), tick=0)


def test_registry_builds_by_name():
    a = build_attack("phantom_actor", distance_m=5.0)
    assert isinstance(a, PhantomActorAttack)
    assert a.params["distance_m"] == 5.0
    with pytest.raises(ValueError):
        build_attack("not_a_real_attack")


# ---------------------------------------------------------------------------
# Shared invariants every attack must satisfy
# ---------------------------------------------------------------------------

ALL_ATTACKS = [ChannelNoiseAttack(), GeometrySpoofAttack(), PhantomActorAttack()]


@pytest.mark.parametrize("attack", ALL_ATTACKS, ids=lambda a: a.name)
@pytest.mark.parametrize("batch", [False, True])
def test_apply_preserves_shape_and_dtype(attack, batch):
    bev, state = make_bev(batch=batch)
    out_bev, out_state = attack.apply(bev, state, tick=5)
    assert out_bev.shape == bev.shape
    assert out_bev.dtype == bev.dtype
    assert out_state.shape == state.shape
    assert out_state.dtype == state.dtype


@pytest.mark.parametrize("attack", ALL_ATTACKS, ids=lambda a: a.name)
def test_apply_does_not_mutate_input_arrays(attack):
    bev, state = make_bev()
    bev_before = bev.copy()
    state_before = state.copy()
    attack.apply(bev, state, tick=5)
    np.testing.assert_array_equal(bev, bev_before)
    np.testing.assert_array_equal(state, state_before)


@pytest.mark.parametrize("attack", ALL_ATTACKS, ids=lambda a: a.name)
def test_apply_output_stays_in_valid_pixel_range(attack):
    bev, state = make_bev()
    out_bev, _ = attack.apply(bev, state, tick=3)
    assert out_bev.min() >= 0 and out_bev.max() <= 255


# ---------------------------------------------------------------------------
# ChannelNoiseAttack specifics
# ---------------------------------------------------------------------------

def test_channel_noise_only_touches_selected_channel():
    bev, state = make_bev()
    attack = ChannelNoiseAttack(channel=DEFAULT_LAYOUT.route, amplitude=100.0)
    out_bev, _ = attack.apply(bev, state, tick=0)
    for ch in range(DEFAULT_LAYOUT.num_channels):
        if ch == DEFAULT_LAYOUT.route:
            continue
        np.testing.assert_array_equal(out_bev[ch], bev[ch])


def test_channel_noise_periodic_mode_oscillates_over_ticks():
    attack = ChannelNoiseAttack(channel=1, amplitude=100.0, frequency_hz=1.0, random_mode=False)
    offsets = [attack._offset(t) for t in range(10)]
    assert max(offsets) > 50.0
    assert min(offsets) < -50.0


def test_channel_noise_random_mode_is_reproducible_with_seed():
    a1 = ChannelNoiseAttack(random_mode=True, seed=42)
    a2 = ChannelNoiseAttack(random_mode=True, seed=42)
    seq1 = [a1._offset(t) for t in range(5)]
    seq2 = [a2._offset(t) for t in range(5)]
    assert seq1 == seq2


# ---------------------------------------------------------------------------
# GeometrySpoofAttack specifics
# ---------------------------------------------------------------------------

def test_geometry_spoof_zero_at_tick_zero():
    bev, state = make_bev()
    attack = GeometrySpoofAttack(max_offset_m=2.0, ramp_ticks=30)
    out_bev, _ = attack.apply(bev, state, tick=0)
    np.testing.assert_array_equal(out_bev[attack.params["channel"]], bev[attack.params["channel"]])


def test_geometry_spoof_reaches_max_offset_after_ramp():
    bev, state = make_bev()
    channel = DEFAULT_LAYOUT.route
    ramp_ticks = 10
    attack = GeometrySpoofAttack(channel=channel, max_offset_m=2.0, ramp_ticks=ramp_ticks)

    out_at_ramp_end, _ = attack.apply(bev, state, tick=ramp_ticks)
    out_well_after, _ = attack.apply(bev, state, tick=ramp_ticks * 5)
    # Offset should be identical (held) once past the ramp.
    np.testing.assert_array_equal(out_at_ramp_end[channel], out_well_after[channel])

    expected = shift_columns(bev[channel], DEFAULT_LAYOUT.meters_to_px(2.0))
    np.testing.assert_array_equal(out_at_ramp_end[channel], expected)


def test_shift_columns_does_not_wrap():
    channel = np.zeros((5, 5), dtype=np.uint8)
    channel[:, 0] = 255  # only the leftmost column is "on"
    shifted = shift_columns(channel, offset_px=2)
    assert shifted[0, 2] == 255
    assert shifted[:, 0].sum() == 0  # vacated edge is zero-filled, not wrapped
    assert shifted[:, -1].sum() == 0  # nothing wrapped around to the right edge


# ---------------------------------------------------------------------------
# PhantomActorAttack specifics
# ---------------------------------------------------------------------------

def test_phantom_actor_absent_before_trigger_tick():
    bev, state = make_bev()
    attack = PhantomActorAttack(trigger_tick=20, distance_m=10.0)
    out_bev, _ = attack.apply(bev, state, tick=5)
    np.testing.assert_array_equal(out_bev, bev)


def test_phantom_actor_appears_in_all_history_slices_once_triggered():
    bev, state = make_bev()
    attack = PhantomActorAttack(trigger_tick=0, distance_m=10.0, lateral_offset_m=0.0, blob_radius_m=1.5)
    out_bev, _ = attack.apply(bev, state, tick=0)

    ego_row, ego_col = DEFAULT_LAYOUT.ego_pixel()
    phantom_row = ego_row - DEFAULT_LAYOUT.meters_to_px(10.0)
    phantom_col = ego_col

    for ch in DEFAULT_LAYOUT.vehicle_channels:
        assert out_bev[ch, phantom_row, phantom_col] == 255, f"channel {ch} missing phantom blob"
    # Should not have touched the walker channels for a vehicle phantom.
    for ch in DEFAULT_LAYOUT.walker_channels:
        np.testing.assert_array_equal(out_bev[ch], bev[ch])


def test_phantom_actor_pedestrian_targets_walker_channels():
    bev, state = make_bev()
    attack = PhantomActorAttack(trigger_tick=0, distance_m=8.0, is_pedestrian=True)
    out_bev, _ = attack.apply(bev, state, tick=0)
    for ch in DEFAULT_LAYOUT.vehicle_channels:
        np.testing.assert_array_equal(out_bev[ch], bev[ch])
    ego_row, ego_col = DEFAULT_LAYOUT.ego_pixel()
    phantom_row = ego_row - DEFAULT_LAYOUT.meters_to_px(8.0)
    for ch in DEFAULT_LAYOUT.walker_channels:
        assert out_bev[ch, phantom_row, ego_col] == 255


def test_draw_disk_clips_gracefully_off_raster():
    channel = np.zeros((10, 10), dtype=np.uint8)
    # Center far outside the array - must not raise or wrap.
    out = draw_disk(channel, center_row=-100, center_col=-100, radius_px=3, value=255)
    assert out.sum() == 0


def test_squeeze_restore_batch_roundtrip():
    bev, state = make_bev(batch=True)
    b, s, had_batch = squeeze_batch(bev, state)
    assert had_batch is True
    assert b.shape == bev.shape[1:]
    rb, rs = restore_batch(b, s, had_batch)
    np.testing.assert_array_equal(rb, bev)
    np.testing.assert_array_equal(rs, state)
