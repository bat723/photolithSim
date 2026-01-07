"""Optical simulation for photolithography."""
import numpy as np
from scipy.special import j1


def airy_psf(shape, na, wavelength_nm, pixel_nm):
    """Generate a 2D Airy disk Point Spread Function."""
    h, w = shape
    y, x = np.indices((h, w))
    y = y - h // 2
    x = x - w // 2
    r_nm = np.hypot(x, y) * pixel_nm
    k = 2 * np.pi * na / wavelength_nm
    v = k * r_nm
    v[v == 0] = 1e-9
    psf = (2 * j1(v) / v) ** 2
    psf = psf / psf.sum()
    return psf


def compute_aerial_image_from_mask(mask, na, wavelength_nm, pixel_nm):
    """Simulate aerial image formation by convolving mask with PSF."""
    from scipy.signal import fftconvolve
    psf = airy_psf(mask.shape, na, wavelength_nm, pixel_nm)
    aerial = fftconvolve(mask, psf, mode='same')
    aerial = aerial / aerial.max() if aerial.max() > 0 else aerial
    return aerial
