"""Phantom actor injection attack.

Draws a fake vehicle or pedestrian blob directly into the BEV raster's
vehicle/walker channels at a controllable distance and lateral offset ahead
of the ego, starting at a trigger tick and persisting for the rest of the
episode. Nothing in the actual CARLA world is touched - no actor is
spawned, ground-truth collision/lane-invasion sensors can't fire on it -
this only tests whether Roach reacts (braking, swerving) to an object that
exists solely in its perceived input.

The blob is drawn into all history slices of the target channel group (not
just the most-recent one), matching docs/attacks.md #2's note that a
single-frame injection looks like a sensor glitch a temporally-aware policy
might discount; a persistent phantom needs to appear consistently across
the history window Roach actually sees.
"""
from __future__ import annotations

from avredteam_carla.attacks.base import Attack, TunableParam
from avredteam_carla.attacks.layout import DEFAULT_LAYOUT
from avredteam_carla.attacks._util import squeeze_batch, restore_batch, draw_disk


class PhantomActorAttack(Attack):
    name = "phantom_actor"
    tunable_params = (
        TunableParam("distance_m", "float", default=15.0, low=1.0, high=30.0),
        TunableParam("lateral_offset_m", "float", default=0.0, low=-10.0, high=10.0),
        TunableParam("trigger_tick", "int", default=0, low=0, high=100_000),
        TunableParam("blob_radius_m", "float", default=1.0, low=0.2, high=5.0),
        TunableParam("is_pedestrian", "bool", default=False),
    )

    def apply(self, bev_raster, scalar_state, tick: int):
        bev, state, had_batch = squeeze_batch(bev_raster, scalar_state)

        if tick < self.params["trigger_tick"]:
            return restore_batch(bev.copy(), state.copy(), had_batch)

        bev = bev.copy()

        ego_row, ego_col = DEFAULT_LAYOUT.ego_pixel()
        # forward = decreasing row index, right = increasing col index
        # (see docs/attacks.md #4 / layout.py's warp-transform derivation).
        phantom_row = ego_row - DEFAULT_LAYOUT.meters_to_px(self.params["distance_m"])
        phantom_col = ego_col + DEFAULT_LAYOUT.meters_to_px(self.params["lateral_offset_m"])
        radius_px = max(1, DEFAULT_LAYOUT.meters_to_px(self.params["blob_radius_m"]))

        channels = (
            DEFAULT_LAYOUT.walker_channels
            if self.params["is_pedestrian"]
            else DEFAULT_LAYOUT.vehicle_channels
        )
        for ch in channels:
            bev[ch] = draw_disk(bev[ch], phantom_row, phantom_col, radius_px, value=255)

        return restore_batch(bev, state.copy(), had_batch)
