"""Geometry spoof attack.

Simulates corrupted map/localization input feeding the rasterizer: the
route/lane channel is gradually shifted sideways (in raster pixel space,
along the column axis - see docs/attacks.md #4/layout.py for why columns
are the lateral axis) over a ramp period, then held at the max offset for
the rest of the episode. Shifting is a zero-fill translation, not a wrap
(np.roll), so it doesn't fabricate content at the opposite raster edge.
"""
from __future__ import annotations

from avredteam_carla.attacks.base import Attack, TunableParam
from avredteam_carla.attacks.layout import DEFAULT_LAYOUT
from avredteam_carla.attacks._util import squeeze_batch, restore_batch, shift_columns


class GeometrySpoofAttack(Attack):
    name = "geometry_spoof"
    tunable_params = (
        TunableParam("channel", "int", default=DEFAULT_LAYOUT.route, low=0, high=DEFAULT_LAYOUT.num_channels - 1),
        # Signed: negative = shift left, positive = shift right.
        TunableParam("max_offset_m", "float", default=2.0, low=-10.0, high=10.0),
        TunableParam("ramp_ticks", "int", default=30, low=1, high=300),
    )

    def apply(self, bev_raster, scalar_state, tick: int):
        bev, state, had_batch = squeeze_batch(bev_raster, scalar_state)
        bev = bev.copy()

        channel = self.params["channel"]
        ramp_ticks = self.params["ramp_ticks"]
        max_offset_m = self.params["max_offset_m"]

        progress = min(1.0, max(0.0, tick / ramp_ticks))
        offset_m = max_offset_m * progress
        offset_px = DEFAULT_LAYOUT.meters_to_px(offset_m)

        bev[channel] = shift_columns(bev[channel], offset_px)

        return restore_batch(bev, state.copy(), had_batch)
