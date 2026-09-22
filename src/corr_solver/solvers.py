"""Reusable cutting-plane solver cores for the correlation oracles.

A *core* drives the ``ellalgo`` cutting-plane loop for one oracle and returns
``(coefficients, num_iters, feasible)``, matching the ``corr_core`` argument of
:func:`~corr_solver.basis.fit`.

The least-squares family augments the coefficient vector with a trailing
objective variable (``x[-1]``); the maximum-likelihood family does not. That
difference is encapsulated by the :class:`Layout` strategies, so the cores no
longer guess dimensions or hardcode the initial ellipsoid values.
"""

from typing import Any, Optional, Protocol, Tuple

import numpy as np
from ellalgo.cutting_plane import BSearchAdaptor, bsearch, cutting_plane_optim
from ellalgo.ell import Ell

from .types import Arr


class Layout(Protocol):
    """Strategy describing the solver's variable vector."""

    def initial(self, Y: Arr, n: int) -> Tuple[Any, Arr]:
        """Return the initial ellipsoid value and point."""
        ...

    def extract(self, x: Arr) -> Arr:
        """Return the coefficients from the solver's variable vector."""
        ...


class MleLayout:
    """Plain coefficient vector of length ``n`` (maximum-likelihood)."""

    def initial(self, Y: Arr, n: int) -> Tuple[Any, Arr]:
        """Return a scalar ellipsoid value and a plain coefficient vector."""
        x = np.zeros(n)
        x[0] = 1.0
        return 50.0, x

    def extract(self, x: Arr) -> Arr:
        """Return the coefficients unchanged."""
        return x


class LSQBsearchLayout:
    """Plain coefficient vector of length ``n`` (least-squares feasibility)."""

    def initial(self, Y: Arr, n: int) -> Tuple[Any, Arr]:
        """Return a scalar ellipsoid value and a plain coefficient vector."""
        x = np.zeros(n)
        x[0] = 1.0
        return 256.0, x

    def extract(self, x: Arr) -> Arr:
        """Return the coefficients unchanged."""
        return x


class LSQAugmentedLayout:
    """Augmented ``(coeffs..., t)`` of length ``n + 1`` (least-squares optimization)."""

    def initial(self, Y: Arr, n: int) -> Tuple[Any, Arr]:
        """Return a per-variable ellipsoid value and an augmented point."""
        normY = np.linalg.norm(Y, "fro")
        normY2 = 32 * normY * normY
        val = 256 * np.ones(n + 1)
        val[-1] = normY2 * normY2
        x = np.zeros(n + 1)
        x[0] = 1.0
        x[-1] = normY2 / 2
        return val, x

    def extract(self, x: Arr) -> Arr:
        """Drop the trailing objective variable."""
        return x[:-1]


def cutting_plane_core(
    Y: Arr, n: int, omega: Any, layout: Layout
) -> Tuple[Optional[Arr], int]:
    """Run ``cutting_plane_optim`` with the given layout.

    :return: ``(coefficients or None, num_iters)``
    """
    val, x = layout.initial(Y, n)
    xbest, _, num_iters = cutting_plane_optim(omega, Ell(val, x), float("inf"))
    if xbest is None:
        return None, num_iters
    return layout.extract(xbest), num_iters


def bsearch_core(Y: Arr, n: int, Q: Any, layout: Layout) -> Tuple[Arr, int, bool]:
    """Run bisection on the objective with the given layout.

    :return: ``(coefficients, num_iters, feasible)``
    """
    val, x = layout.initial(Y, n)
    omega = BSearchAdaptor(Q, Ell(val, x))
    upper = np.linalg.norm(Y, "fro") ** 2
    t, num_iters = bsearch(omega, (0.0, upper))
    return layout.extract(omega.x_best), num_iters, t != upper


def lsq_corr_core(Y: Arr, n: int, Q: Any) -> Tuple[Arr, int, bool]:
    """Least-squares core: bisection on the objective value.

    :param Y: biased sample covariance matrix
    :param n: number of coefficients
    :param Q: feasibility oracle
    :return: ``(coefficients, num_iters, feasible)``
    """
    return bsearch_core(Y, n, Q, LSQBsearchLayout())


def lsq_corr_core2(Y: Arr, n: int, omega: Any) -> Tuple[Arr, int, bool]:
    """Least-squares core: augmented ``(x, t)`` cutting-plane optimization.

    :param Y: biased sample covariance matrix
    :param n: number of coefficients
    :param omega: optimization oracle
    :return: ``(coefficients, num_iters, feasible)``
    """
    coeffs, num_iters = cutting_plane_core(Y, n, omega, LSQAugmentedLayout())
    if coeffs is None:
        return np.zeros(n), num_iters, False
    return coeffs, num_iters, True


def mle_corr_core(Y: Arr, n: int, omega: Any) -> Tuple[Optional[Arr], int, bool]:
    """Maximum-likelihood core.

    :param Y: biased sample covariance matrix (unused; kept for the core signature)
    :param n: number of coefficients
    :param omega: optimization oracle
    :return: ``(coefficients, num_iters, feasible)``
    """
    coeffs, num_iters = cutting_plane_core(Y, n, omega, MleLayout())
    return coeffs, num_iters, coeffs is not None
