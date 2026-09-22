"""Radial covariance-kernel strategies.

Every kernel maps a squared distance ``r2`` and a ``rate`` to a covariance value:
``rate`` is the exponential decay rate for the Gaussian and the inverse length
scale for the Matern family. ``KERNELS`` is the name-to-kernel factory.
"""

from typing import Callable, Dict

import numpy as np

from .types import Arr


def gaussian(r2: Arr, rate: float) -> Arr:
    """``exp(-rate * r2)``."""
    return np.exp(-rate * r2)


def exponential(r2: Arr, rate: float) -> Arr:
    """Matern 1/2: ``exp(-rate * r)``."""
    return np.exp(-rate * np.sqrt(r2))


def matern32(r2: Arr, rate: float) -> Arr:
    """Matern 3/2: ``(1 + rate r) exp(-rate r)``."""
    z = rate * np.sqrt(r2)
    return np.exp(np.log1p(z) - z)


def matern52(r2: Arr, rate: float) -> Arr:
    """Matern 5/2: ``(1 + rate r + (rate r)^2 / 3) exp(-rate r)``."""
    z = rate * np.sqrt(r2)
    return np.exp(np.log1p(z + z**2 / 3.0) - z)


KERNELS: Dict[str, Callable[[Arr, float], Arr]] = {
    "gaussian": gaussian,
    "exponential": exponential,
    "matern32": matern32,
    "matern52": matern52,
}
