"""
Correlation oracles.

Building blocks for fitting correlation/error-correction polynomials to biased
covariance matrices:

1. ``create_2d_sites`` places sites on a 2D grid using a Halton sequence.
2. ``create_2d_isotropic`` generates a biased sample covariance matrix from an
   isotropic Gaussian kernel over the site distances.
3. ``corr_poly`` fits a polynomial to ``Y`` via a cutting-plane oracle and
   returns a ``poly1d`` object, the iteration count, and a feasibility flag.

``construct_distance_matrix`` and ``construct_poly_matrix`` live in
:mod:`corr_solver.basis` and are re-exported here for backward compatibility.
"""

from typing import Any

import numpy as np
from lds_gen.lds import Halton
from scipy.spatial.distance import pdist, squareform

from .basis import (
    PolynomialBasis,
    construct_distance_matrix,
    construct_poly_matrix,
    fit,
)
from .kernels import gaussian
from .types import Arr, FitResult

__all__ = [
    "create_2d_sites",
    "create_2d_isotropic",
    "construct_distance_matrix",
    "construct_poly_matrix",
    "corr_poly",
]


def create_2d_sites(nx: int = 10, ny: int = 8) -> Arr:
    """
    The function `create_2d_sites` generates a 2D array of site locations using the Halton sequence.

    :param nx: The parameter `nx` represents the number of sites in the x-direction, while `ny`
        represents the number of sites in the y-direction, defaults to 10 (optional)
    :param ny: The parameter `ny` represents the number of rows in the 2D sites object, defaults to 8
        (optional)
    :return: The function `create_2d_sites` returns a 2D array representing the location of sites.
    """
    num_grid = nx * ny
    s_end = np.array([10.0, 8.0])
    hgen = Halton([2, 3])
    site = s_end * np.array([hgen.pop() for _ in range(num_grid)])
    return site


def create_2d_isotropic(site: Arr, N: int = 3000, rng: Any = None) -> Arr:
    """
    The function `create_2d_isotropic` generates a biased covariance matrix for a 2D isotropic object
    based on the location of sites.

    :param site: The parameter `site` is the location of sites. It is expected to be a 2D array where each row
        represents the coordinates of a site
    :type site: Arr
    :param N: The parameter N represents the number of iterations or samples used to create the 2D
        isotropic object. It determines the number of times the loop runs to generate random values and
        calculate the outer product. The larger the value of N, the more accurate the estimation of the
        biased covariance matrix will be,, defaults to 3000 (optional)
    :param rng: optional random source; defaults to ``RandomState(5)`` for
        reproducible output
    :return: The function `create_2d_isotropic` returns a biased covariance matrix `Y`.
    """
    n = site.shape[0]
    sdkern = 0.12  # width of kernel
    var = 2.0  # standard derivation
    tau = 0.00001  # standard derivation of white noise
    if rng is None:
        rng = np.random.RandomState(5)

    # Vectorized covariance construction via pairwise squared distances
    dist_sq = squareform(pdist(site, "sqeuclidean"))
    Sigma = gaussian(dist_sq, sdkern)

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


def corr_poly(Y: Arr, site: Arr, m: int, oracle: Any, corr_core: Any) -> FitResult:
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
    :return: The function `corr_poly` returns a tuple containing three elements:
        1. A polynomial object representing the polynomial fit to the data.
        2. The number of iterations performed during the correction process.
        3. A boolean value indicating whether a feasible solution was found.
    """
    return fit(Y, site, m, oracle, corr_core, PolynomialBasis())
