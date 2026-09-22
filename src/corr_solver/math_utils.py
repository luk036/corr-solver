"""Shared linear-algebra helpers for the correlation oracles.

These are the pieces that were previously copied between the pure-Python and
Numba oracles. The arithmetic order is kept exactly as it was in those call
sites: the quadratic matrix inequality is evaluated at the boundary of the
positive-semidefinite cone, where the first pivot to turn negative is decided
by rounding, so reordering the sums can pick a different separating cut.
"""

from typing import List, Tuple

import numpy as np

from .types import Arr


def omega_of(x: Arr, Sigma: List[Arr]) -> Arr:
    """Return ``Omega(x) = sum_i x_i Sigma_i``.

    :param x: coefficient vector
    :param Sigma: basis matrices
    :return: the assembled matrix
    """
    return sum(c * F for c, F in zip(x, Sigma))


def mle_obj(x: Arr, Sigma: List[Arr], Y: Arr) -> float:
    """Return ``log det Omega(x) + Tr(Omega(x)^-1 Y)``, or ``inf`` if not positive definite.

    :param x: coefficient vector
    :param Sigma: basis matrices
    :param Y: biased sample covariance matrix
    :return: the MLE objective value
    """
    Om = omega_of(x, Sigma)
    sign, logdet = np.linalg.slogdet(Om)
    return np.inf if sign <= 0 else float(logdet + np.trace(np.linalg.solve(Om, Y)))


def inverse_sqrt_gram(R: Arr, Y: Arr) -> Tuple[Arr, Arr]:
    """Return ``S = (R^-1)(R^-1)^T`` and ``S @ Y``.

    :param R: upper-triangular factor with ``Omega = R^T R``
    :param Y: biased sample covariance matrix
    :return: the pair ``(S, S @ Y)``
    """
    invR = np.linalg.inv(R)
    S = invR @ invR.T
    return S, S @ Y


def mle_value_and_grad(R: Arr, Y: Arr, Sigma: List[Arr]) -> Tuple[float, Arr]:
    """Return the MLE objective ``2 sum log diag(R) + Tr(SY)`` and its gradient.

    :param R: upper-triangular factor with ``Omega = R^T R``
    :param Y: biased sample covariance matrix
    :param Sigma: basis matrices
    :return: the pair ``(f1, g)``
    """
    S, SY = inverse_sqrt_gram(R, Y)
    f1 = 2 * np.sum(np.log(np.diag(R))) + np.trace(SY)
    # g[i] = tr(S @ Sigma[i]) - tr(S @ Sigma[i] @ SY)
    #      = tr((S - SY @ S) @ Sigma[i])      [cyclic perm of trace]
    #      = sum((S - SY @ S).T * Sigma[i])    [Frobenius inner prod]
    V = S - SY @ S  # pre-compute once, n×n
    g = np.array([np.sum(V.T * F) for F in Sigma])
    return float(f1), g
