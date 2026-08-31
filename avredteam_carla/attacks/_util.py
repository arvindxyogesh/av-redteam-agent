"""Shared array helpers for attack implementations. Pure numpy - no cv2/scipy
dependency, so these are testable without the Roach conda env installed."""
from __future__ import annotations

import numpy as np

# CARLA runs in synchronous mode with fixed_delta_seconds=0.1 (see
# carla_multi_agent_env.py's set_sync_mode, and Phase 1's docs/setup.md) -
# i.e. 10 ticks per simulated second, always. Attacks that need a frequency
# in Hz convert against this fixed rate.
SIM_HZ = 10.0


def squeeze_batch(bev: np.ndarray, state: np.ndarray):
    """Eval-time tensors carry a leading batch dim of 1 (see docs/attacks.md
    #3). Returns (bev_no_batch, state_no_batch, had_batch)."""
    had_batch = bev.ndim == 4
    if had_batch:
        assert bev.shape[0] == 1 and state.shape[0] == 1, (
            f"expected batch size 1, got bev {bev.shape} state {state.shape}"
        )
        return bev[0], state[0], True
    return bev, state, False


def restore_batch(bev: np.ndarray, state: np.ndarray, had_batch: bool):
    if had_batch:
        return bev[np.newaxis], state[np.newaxis]
    return bev, state


def shift_columns(channel: np.ndarray, offset_px: int, fill: int = 0) -> np.ndarray:
    """Shift a (H, W) channel sideways by offset_px columns (positive =
    right), filling the vacated edge with `fill` rather than wrapping
    around (np.roll would wrap, which fabricates content at the opposite
    edge of the raster - a bigger deviation from the source image than a
    lateral bias is meant to introduce)."""
    if offset_px == 0:
        return channel.copy()
    shifted = np.full_like(channel, fill)
    if offset_px > 0:
        if offset_px < channel.shape[1]:
            shifted[:, offset_px:] = channel[:, : channel.shape[1] - offset_px]
    else:
        if -offset_px < channel.shape[1]:
            shifted[:, :offset_px] = channel[:, -offset_px:]
    return shifted


def draw_disk(channel: np.ndarray, center_row: int, center_col: int, radius_px: int, value: int = 255) -> np.ndarray:
    """Draw a filled disk into a copy of a (H, W) channel, clipped to bounds.
    Pure numpy (no cv2) so this is usable without the Roach conda env."""
    out = channel.copy()
    h, w = out.shape
    row0, row1 = max(0, center_row - radius_px), min(h, center_row + radius_px + 1)
    col0, col1 = max(0, center_col - radius_px), min(w, center_col + radius_px + 1)
    if row0 >= row1 or col0 >= col1:
        return out  # entirely off-raster, nothing to draw
    rows, cols = np.ogrid[row0:row1, col0:col1]
    mask = (rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius_px ** 2
    region = out[row0:row1, col0:col1]
    region[mask] = value
    out[row0:row1, col0:col1] = region
    return out
