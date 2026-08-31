"""Unit tests for masks_to_rgb (pure numpy, no cv2/PIL needed - save_png
does need one of those and isn't covered here; see docs/attacks.md #5)."""
import numpy as np

from avredteam_carla.attacks.layout import DEFAULT_LAYOUT
from avredteam_carla.attacks.visualize import masks_to_rgb, ROUTE_COLOR, VEHICLE_COLOR, EGO_COLOR


def test_masks_to_rgb_shape_and_dtype():
    masks = np.zeros((DEFAULT_LAYOUT.num_channels, 192, 192), dtype=np.uint8)
    out = masks_to_rgb(masks)
    assert out.shape == (192, 192, 3)
    assert out.dtype == np.uint8


def test_masks_to_rgb_paints_route_pixels():
    masks = np.zeros((DEFAULT_LAYOUT.num_channels, 192, 192), dtype=np.uint8)
    masks[DEFAULT_LAYOUT.route, 50, 60] = 255
    out = masks_to_rgb(masks)
    assert tuple(out[50, 60]) == ROUTE_COLOR


def test_masks_to_rgb_vehicle_drawn_over_route():
    masks = np.zeros((DEFAULT_LAYOUT.num_channels, 192, 192), dtype=np.uint8)
    masks[DEFAULT_LAYOUT.route, 50, 60] = 255
    masks[DEFAULT_LAYOUT.vehicle_latest, 50, 60] = 255
    out = masks_to_rgb(masks)
    assert tuple(out[50, 60]) == VEHICLE_COLOR  # later layer wins


def test_masks_to_rgb_draws_ego_marker():
    masks = np.zeros((DEFAULT_LAYOUT.num_channels, 192, 192), dtype=np.uint8)
    out = masks_to_rgb(masks)
    ego_row, ego_col = DEFAULT_LAYOUT.ego_pixel()
    assert tuple(out[ego_row, ego_col]) == EGO_COLOR


def test_masks_to_rgb_rejects_batched_input():
    masks = np.zeros((1, DEFAULT_LAYOUT.num_channels, 192, 192), dtype=np.uint8)
    try:
        masks_to_rgb(masks)
        assert False, "expected ValueError for a batched (4D) array"
    except ValueError:
        pass
