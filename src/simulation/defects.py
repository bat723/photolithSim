"""Defect injection functions."""
import numpy as np
from skimage.draw import disk


def add_particle_defect(image, particle_radius_px=5, intensity=0.8):
    """Add a bright particle (contamination)."""
    img_defect = image.copy()
    h, w = img_defect.shape
    margin = particle_radius_px + 2
    cy = np.random.randint(margin, h - margin)
    cx = np.random.randint(margin, w - margin)
    rr, cc = disk((cy, cx), particle_radius_px, shape=img_defect.shape)
    img_defect[rr, cc] = np.clip(img_defect[rr, cc] + intensity, 0, 1)
    return img_defect


def add_line_break(mask, break_length_px=20):
    """Create a missing segment in a line."""
    mask_defect = mask.copy()
    h, w = mask_defect.shape
    y = h // 2
    x_start = w // 4
    mask_defect[y-2:y+2, x_start:x_start+break_length_px] = 0
    return mask_defect


def add_line_roughness(image, roughness_amplitude=0.05):
    """Add edge roughness (LER)."""
    noise = np.random.normal(0, roughness_amplitude, image.shape)
    return np.clip(image + noise, 0, 1)

