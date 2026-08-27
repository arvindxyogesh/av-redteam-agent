"""Channel/geometry layout of Roach's BEV raster, as documented in
docs/attacks.md #2 (read from carla_gym's chauffeurnet.py ObsManager).

Kept as a small dataclass rather than hardcoded indices everywhere so an
attack's channel targeting stays correct if the deployed obs_configs ever
changes history_idx length (Phase 1/2 use the repo default, 4 entries).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class BirdviewLayout:
    width_px: int = 192
    pixels_per_meter: float = 5.0
    pixels_ev_to_bottom: int = 40
    history_len: int = 4  # len(history_idx), e.g. [-16, -11, -6, -1]

    @property
    def num_channels(self) -> int:
        return 3 + 3 * self.history_len

    @property
    def road(self) -> int:
        return 0

    @property
    def route(self) -> int:
        return 1

    @property
    def lane(self) -> int:
        return 2

    @property
    def vehicle_channels(self) -> range:
        start = 3
        return range(start, start + self.history_len)

    @property
    def walker_channels(self) -> range:
        start = 3 + self.history_len
        return range(start, start + self.history_len)

    @property
    def traffic_light_channels(self) -> range:
        start = 3 + 2 * self.history_len
        return range(start, start + self.history_len)

    @property
    def vehicle_latest(self) -> int:
        """Most recent (t-1) vehicle-history channel - see docs/attacks.md #2."""
        return self.vehicle_channels[-1]

    @property
    def walker_latest(self) -> int:
        return self.walker_channels[-1]

    def ego_pixel(self) -> tuple:
        """(row, col) of the ego position within the raster.

        Per chauffeurnet.py's warp transform, the ego is horizontally
        centered and pixels_ev_to_bottom px up from the bottom edge (row
        indices increase downward in the array, so "up from the bottom" is
        width_px - pixels_ev_to_bottom).
        """
        row = self.width_px - self.pixels_ev_to_bottom
        col = self.width_px // 2
        return row, col

    def meters_to_px(self, meters: float) -> int:
        return int(round(meters * self.pixels_per_meter))


DEFAULT_LAYOUT = BirdviewLayout()
