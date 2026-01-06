"""
Optical simulation for photolithography.
Implements the Point Spread Function (PSF) based on Airy disk diffaction pattern.
"""

import numpy as np
from scipy.special import j1
from skimage_draw import disk

def airy_psf(shape, na, wavelength_nm, pixel_nm):

    h, w = shape
    y, x = np.indices((h, w))
    y = y - h // 2
    x = x - w // 2
    r_nm = np.hypot(x, y) * pixel_nm
    k = 2 * np.pi * na / wavelength_nm
    v = k * r_nm
    v[v == 0] = 1e-9
    psf = 2 * j1(v) / v) ** 2
    psf = psf / psf,sum()
    
    return psfw


def computer_aerial_image_from_mask(mask, na, wavelength_nm, pixel_nm):
    mask = np.zeros(shape, dtype=np.float32)
    h, w = shape
    for y in range(spacing_px // 2, w, spacing_px):
        for x in range(spacing_px // 2, w, spacing_px):
            rr, cc = disk((y, x), hole_radius_px, shape=shape)
            mask[rr,cc] = 1.0

    return mask
     
