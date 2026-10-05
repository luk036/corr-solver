"""
Correlation oracles.

Building blocks for fitting correlation/error-correction polynomials to biased
covariance matrices:

1. ``create_2d_sites`` places sites on a 2D grid using a Halton sequence.
2. ``_sample_covariance`` averages Gaussian samples for a kernel matrix, and
   ``create_2d_isotropic`` / ``create_2d_anisotropic`` build that kernel from the
   site distances (Euclidean or per-axis scaled) via the ``KERNELS`` registry.
3. ``corr_poly`` fits a polynomial to ``Y`` via a cutting-plane oracle and
   returns a ``poly1d`` object, the iteration count, and a feasibility flag.

``construct_distance_matrix`` and ``construct_poly_matrix`` live in
:mod:`corr_solver.basis` and are re-exported here for backward compatibility.
"""

from typing import Any, Optional

import numpy as np
from lds_gen.lds import Halton
from scipy.spatial.distance import pdist, squareform

from .basis import (
    PolynomialBasis,
    construct_distance_matrix,
    construct_poly_matrix,
    fit,
)
from .kernels import KERNELS
from .solvers import SolverConfig
from .types import Arr, FitResult

__all__ = [
    "create_2d_sites",
    "create_2d_isotropic",
    "create_2d_anisotropic",
    "construct_distance_matrix",
    "construct_poly_matrix",
    "corr_poly",
]


def create_2d_sites(nx: int = 5, ny: int = 4) -> Arr:
    """
    The function `create_2d_sites` generates a 2D array of site locations using the Halton sequence.

    :param nx: The parameter `nx` represents the number of sites in the x-direction, while `ny`
        represents the number of sites in the y-direction, defaults to 5 (optional)
    :param ny: The parameter `ny` represents the number of rows in the 2D sites object, defaults to 4
        (optional)
    :return: The function `create_2d_sites` returns a 2D array representing the location of sites.
    """
    num_grid = nx * ny
    s_end = np.array([10.0, 8.0])
    hgen = Halton([2, 3])
    site = s_end * np.array([hgen.pop() for _ in range(num_grid)])
    return site


def _sample_covariance(Sigma: Arr, N: int, rng: Any, var: float, tau: float) -> Arr:
    """Average ``N`` draws of ``y y^T`` with ``y ~ N(0, var^2 Sigma + tau^2 I)``."""
    n = Sigma.shape[0]
    A = np.linalg.cholesky(Sigma)
    Y = np.zeros((n, n))
    outer_buf = np.empty((n, n))
    for _ in range(N):
        x = var * rng.randn(n)
        y = A @ x + tau * rng.randn(n)
        np.outer(y, y, out=outer_buf)
        Y += outer_buf
    Y /= N
    return Y


def _sample_from_dist_sq(
    dist_sq: Arr,
    N: int,
    rng: Any,
    kernel: str,
    rate: float,
    var: float = 2.0,
    tau: float = 1e-5,
) -> Arr:
    """Kernel the squared distances and sample a covariance from the result."""
    if rng is None:
        rng = np.random.RandomState(5)
    Sigma = KERNELS[kernel](dist_sq, rate)
    return _sample_covariance(Sigma, N, rng, var=var, tau=tau)


def create_2d_isotropic(
    site: Arr,
    N: int = 1000,
    rng: Any = None,
    kernel: str = "gaussian",
    rate: float = 0.12,
    var: float = 2.0,
    tau: float = 1e-5,
) -> Arr:
    """
    The function `create_2d_isotropic` generates a biased covariance matrix for a 2D isotropic object
    based on the location of sites.

    :param site: site locations, one row per site
    :type site: Arr
    :param N: number of samples averaged to estimate the covariance, defaults to 1000
    :param rng: optional random source; defaults to ``RandomState(5)`` for
        reproducible output
    :param kernel: a key of :data:`corr_solver.kernels.KERNELS`
    :param rate: the kernel rate / inverse length scale
    :param var: signal standard deviation
    :param tau: observation-noise standard deviation
    :return: a biased sample covariance matrix `Y`.
    """
    dist_sq = squareform(pdist(site, "sqeuclidean"))
    return _sample_from_dist_sq(dist_sq, N, rng, kernel, rate, var, tau)


def create_2d_anisotropic(
    site: Arr,
    length_x: float,
    length_y: float,
    N: int = 1000,
    rng: Any = None,
    kernel: str = "gaussian",
    rate: float = 0.12,
    var: float = 2.0,
    tau: float = 1e-5,
) -> Arr:
    """Biased sample covariance from a kernel with per-axis length scales.

    :param site: site locations, one row per site
    :param length_x: length scale along the first coordinate
    :param length_y: length scale along the second coordinate
    :param N: number of samples averaged to estimate the covariance
    :param rng: optional random source; defaults to ``RandomState(5)``
    :param kernel: a key of :data:`corr_solver.kernels.KERNELS`
    :param rate: the kernel rate / inverse length scale
    :param var: signal standard deviation
    :param tau: observation-noise standard deviation
    :return: a biased sample covariance matrix `Y`.
    """
    dx = site[:, None, 0] - site[None, :, 0]
    dy = site[:, None, 1] - site[None, :, 1]
    dist_sq = (dx / length_x) ** 2 + (dy / length_y) ** 2
    return _sample_from_dist_sq(dist_sq, N, rng, kernel, rate, var, tau)


def corr_poly(
    Y: Arr,
    site: Arr,
    m: int,
    oracle: Any,
    corr_core: Any,
    config: Optional[SolverConfig] = None,
) -> FitResult:
    """
    The function `corr_poly` takes in a signal `Y`, a sparsity level `site`, a maximum degree `m`, an
    oracle function, and a correction core function, and returns a polynomial, the number of iterations,
    and a feasibility indicator.

    :param Y: The parameter `Y` represents the input data, which is a vector or matrix of shape
        (n_samples, n_features). It contains the input variables for which we want to find a polynomial
        correlation
    :param site: The parameter `site` represents the degree of the polynomial. It determines the number of
        coefficients in the polynomial
    :param m: The parameter `m` represents the degree of the polynomial that you want to construct. It
        determines the number of coefficients in the polynomial
    :param oracle: The `oracle` parameter is a function that takes in two arguments: `Sigma` and `Y`.
        `Sigma` is a matrix and `Y` is a vector. The `oracle` function returns a vector `omega`
    :param corr_core: The `corr_core` parameter is a function that takes in the following arguments:
    :param config: optional :class:`~corr_solver.solvers.SolverConfig`; defaults to
        the solver core's own configuration
    :return: The function `corr_poly` returns a tuple containing three elements:
        1. A polynomial object representing the polynomial fit to the data.
        2. The number of iterations performed during the correction process.
        3. A boolean value indicating whether a feasible solution was found.
    """
    return fit(Y, site, m, oracle, corr_core, PolynomialBasis(), config)
