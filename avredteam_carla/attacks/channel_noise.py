"""Channel noise / oscillation attack.

Adds a periodic (sinusoidal) or random additive offset to every pixel of one
BEV channel, each tick. For a near-binary channel like route/lane (values
{0,255} or {0,120,255} - see docs/attacks.md #2), a uniform additive offset
degrades that channel's on/off contrast asymmetrically depending on sign:
a negative offset dims the "on" (255) pixels toward the background while
leaving the "off" (0) background at 0 (clipped); a positive offset raises
the background toward the "on" level while "on" pixels stay saturated.
Oscillating the offset (periodic mode) therefore oscillates the route/lane
channel's signal-to-noise ratio over time - the intended analogue to
inducing a periodic steering response.
"""
from __future__ import annotations

import numpy as np

from avredteam_carla.attacks.base import Attack, TunableParam
from avredteam_carla.attacks.layout import DEFAULT_LAYOUT
from avredteam_carla.attacks._util import SIM_HZ, squeeze_batch, restore_batch


class ChannelNoiseAttack(Attack):
    name = "channel_noise"
    tunable_params = (
        TunableParam("channel", "int", default=DEFAULT_LAYOUT.route, low=0, high=DEFAULT_LAYOUT.num_channels - 1),
        TunableParam("amplitude", "float", default=80.0, low=0.0, high=255.0),
        TunableParam("frequency_hz", "float", default=2.0, low=0.05, high=5.0),
        TunableParam("random_mode", "bool", default=False),
        TunableParam("seed", "int", default=0, low=0, high=2**31 - 1),
    )

    def __init__(self, **param_overrides):
        super().__init__(**param_overrides)
        self._rng = np.random.default_rng(self.params["seed"])

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.params["seed"])

    def _offset(self, tick: int) -> float:
        amplitude = self.params["amplitude"]
        if self.params["random_mode"]:
            return float(self._rng.standard_normal()) * amplitude
        freq = self.params["frequency_hz"]
        phase = 2.0 * np.pi * freq * (tick / SIM_HZ)
        return amplitude * np.sin(phase)

    def apply(self, bev_raster: np.ndarray, scalar_state: np.ndarray, tick: int):
        bev, state, had_batch = squeeze_batch(bev_raster, scalar_state)
        bev = bev.copy()

        channel = self.params["channel"]
        offset = self._offset(tick)

        perturbed = bev[channel].astype(np.int16) + offset
        bev[channel] = np.clip(perturbed, 0, 255).astype(np.uint8)

        return restore_batch(bev, state.copy(), had_batch)
