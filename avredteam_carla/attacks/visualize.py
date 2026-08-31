"""Renders Roach's actual policy-input BEV tensor (the 'masks' array, not
carla_gym's own 'rendered' image) into a human-viewable RGB picture.

This exists because carla_gym's 'rendered' image is computed independently
of 'masks' inside ObsManager.get_observation() (see docs/attacks.md #1) -
it's produced *before* our hook ever runs, so it never reflects what an
attack did. For a visual sanity check of an attack's actual effect, we need
to colorize the post-attack 'masks' tensor itself. Deliberately simpler than
chauffeurnet.py's own renderer (no traffic-light tinting by recency) - this
only needs to be legible enough to eyeball "did the attack do what it should"
on a headless cluster, not publication-quality.
"""
from __future__ import annotations

import numpy as np

from avredteam_carla.attacks.layout import BirdviewLayout, DEFAULT_LAYOUT

ROAD_COLOR = (46, 52, 54)
ROUTE_COLOR = (136, 138, 133)
LANE_COLOR = (255, 0, 255)
VEHICLE_COLOR = (0, 0, 255)
WALKER_COLOR = (0, 255, 255)
EGO_COLOR = (255, 255, 255)
EGO_MARKER_RADIUS_PX = 3


def masks_to_rgb(masks: np.ndarray, layout: BirdviewLayout = DEFAULT_LAYOUT) -> np.ndarray:
    """masks: (C, H, W) uint8, no batch dim (squeeze first if needed).
    Returns (H, W, 3) uint8, paintable in this layer order: road, route,
    lane, most-recent walker, most-recent vehicle, ego marker - later layers
    drawn on top of earlier ones, same precedence chauffeurnet.py uses."""
    if masks.ndim != 3:
        raise ValueError(f"expected (C, H, W), got shape {masks.shape}")

    h, w = masks.shape[1], masks.shape[2]
    image = np.zeros((h, w, 3), dtype=np.uint8)

    image[masks[layout.road] > 0] = ROAD_COLOR
    image[masks[layout.route] > 0] = ROUTE_COLOR
    image[masks[layout.lane] > 0] = LANE_COLOR
    image[masks[layout.walker_latest] > 0] = WALKER_COLOR
    image[masks[layout.vehicle_latest] > 0] = VEHICLE_COLOR

    ego_row, ego_col = layout.ego_pixel()
    r0, r1 = max(0, ego_row - EGO_MARKER_RADIUS_PX), min(h, ego_row + EGO_MARKER_RADIUS_PX + 1)
    c0, c1 = max(0, ego_col - EGO_MARKER_RADIUS_PX), min(w, ego_col + EGO_MARKER_RADIUS_PX + 1)
    image[r0:r1, c0:c1] = EGO_COLOR

    return image


def save_png(rgb: np.ndarray, path) -> None:
    """Writes an (H, W, 3) uint8 array as a PNG without adding a new
    dependency: prefers cv2 (already required by carla_gym itself, so it's
    present in Roach's env), falls back to Pillow if cv2 isn't importable
    (e.g. in a plain dev sandbox with neither installed - see the Phase 2
    PR for how this was actually exercised on Maui)."""
    try:
        import cv2

        # cv2 expects BGR.
        cv2.imwrite(str(path), rgb[:, :, ::-1])
        return
    except ImportError:
        pass

    from PIL import Image

    Image.fromarray(rgb, mode="RGB").save(str(path))
