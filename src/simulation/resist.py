"""Resist dissolution & contrast-curve utilities."""
import numpy as np

def remaining_resist_fraction(aerial, dose_mj, gamma=2.5, dose_to_clear=20.0):
    """
    Simplified positive-resist model:
    - aerial: float image in [0,1]
    - dose_mj: exposure dose (mJ/cm^2)
    - gamma: contrast-like exponent (higher = steeper curve)
    - dose_to_clear: characteristic dose where fully exposed areas clear
    Returns: remaining resist fraction in [0,1]
    """
    E = aerial * dose_mj
    removed = (E / dose_to_clear) ** gamma
    removed = np.clip(removed, 0.0, 1.0)
    return 1.0 - removed
