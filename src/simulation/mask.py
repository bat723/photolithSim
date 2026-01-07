"""Photomask pattern generators."""
import numpy as np
from skimage.draw import disk


def create_lines_and_spaces(shape=(256, 256), pitch_px=32, duty_cycle=0.5):
    """Create a simple line-space grating pattern."""
    h, w = shape
    line_width = int(pitch_px * duty_cycle)
    space_width = pitch_px - line_width
    period = np.concatenate([np.ones(line_width), np.zeros(space_width)])
    pattern = np.tile(period, w // pitch_px + 1)[:w]
    mask = np.tile(pattern, (h, 1))
    return mask.astype(np.float32)


def create_contact_holes(shape=(256, 256), spacing_px=40, hole_radius_px=8):
    """Create a 2D array of circular contact holes."""
    mask = np.zeros(shape, dtype=np.float32)
    h, w = shape
    for y in range(spacing_px // 2, h, spacing_px):
        for x in range(spacing_px // 2, w, spacing_px):
            rr, cc = disk((y, x), hole_radius_px, shape=shape)
            mask[rr, cc] = 1.0
    return mask
