"""Reusable cutting-plane solver cores for the correlation oracles.

A *core* drives the ``ellalgo`` cutting-plane loop for one oracle and returns
``(coefficients, num_iters, feasible)``, matching the ``corr_core`` argument of
:func:`~corr_solver.basis.fit`.

The least-squares family augments the coefficient vector with a trailing
objective variable (``x[-1]``); the maximum-likelihood family does not. That
difference is encapsulated by the :class:`Layout` strategies, so the cores no
longer guess dimensions or hardcode the initial ellipsoid values.
"""

from dataclasses import dataclass
from typing import Any, Protocol, Tuple, Union

import numpy as np
from ellalgo.cutting_plane import BSearchAdaptor, bsearch, cutting_plane_optim
from ellalgo.ell import Ell

from .types import Arr, CoreResult


@dataclass(frozen=True)
class SolverConfig:
    """Numeric constants for the cutting-plane and bisection cores.

    :param mle_r0: initial MLE ellipsoid scale
    :param lsq_bs_r0: initial ellipsoid scale for the bisection feasibility core
    :param lsq_aug_r0: initial ellipsoid scale for the augmented LSQ core
    :param lsq_frob_scale: multiplier on ``||Y||_F^2`` for the augmented bound
    :param cccp_r0: initial ellipsoid scale for the CCP outer loop
    """

    mle_r0: float = 50.0
    lsq_bs_r0: float = 256.0
    lsq_aug_r0: float = 256.0
    lsq_frob_scale: float = 32.0
    cccp_r0: float = 100.0


class Layout(Protocol):
    """Strategy describing the solver's variable vector."""

    def initial(self, Y: Arr, n: int) -> Tuple[Union[float, Arr], Arr]:
        """Return the initial ellipsoid value and point."""
        ...

    def extract(self, x: Arr) -> Arr:
        """Return the coefficients from the solver's variable vector."""
        ...


class MleLayout:
    """Plain coefficient vector of length ``n`` (maximum-likelihood)."""

    def __init__(self, config: SolverConfig = SolverConfig()) -> None:
        self.config = config

    def initial(self, Y: Arr, n: int) -> Tuple[Union[float, Arr], Arr]:
        """Return a scalar ellipsoid value and a plain coefficient vector."""
        x = np.zeros(n)
        x[0] = 1.0
        return self.config.mle_r0, x

    def extract(self, x: Arr) -> Arr:
        """Return the coefficients unchanged."""
        return x


class LSQBsearchLayout:
    """Plain coefficient vector of length ``n`` (least-squares feasibility)."""

    def __init__(self, config: SolverConfig = SolverConfig()) -> None:
        self.config = config

    def initial(self, Y: Arr, n: int) -> Tuple[Union[float, Arr], Arr]:
        """Return a scalar ellipsoid value and a plain coefficient vector."""
        x = np.zeros(n)
        x[0] = 1.0
        return self.config.lsq_bs_r0, x

    def extract(self, x: Arr) -> Arr:
        """Return the coefficients unchanged."""
        return x


class LSQAugmentedLayout:
    """Augmented ``(coeffs..., t)`` of length ``n + 1`` (least-squares optimization)."""

    def __init__(self, config: SolverConfig = SolverConfig()) -> None:
        self.config = config

    def initial(self, Y: Arr, n: int) -> Tuple[Union[float, Arr], Arr]:
        """Return a per-variable ellipsoid value and an augmented point."""
        normY = np.linalg.norm(Y, "fro")
        normY2 = self.config.lsq_frob_scale * normY * normY
        val = self.config.lsq_aug_r0 * np.ones(n + 1)
        val[-1] = normY2 * normY2
        x = np.zeros(n + 1)
        x[0] = 1.0
        x[-1] = normY2 / 2
        return val, x

    def extract(self, x: Arr) -> Arr:
        """Drop the trailing objective variable."""
        return x[:-1]


def cutting_plane_core(Y: Arr, n: int, omega: Any, layout: Layout) -> CoreResult:
    """Run ``cutting_plane_optim`` with the given layout.

    :return: a :class:`~corr_solver.types.CoreResult`
    """
    val, x = layout.initial(Y, n)
    xbest, _, num_iters = cutting_plane_optim(omega, Ell(val, x), float("inf"))
    if xbest is None:
        return CoreResult(None, num_iters, False)
    return CoreResult(layout.extract(xbest), num_iters, True)


def bsearch_core(Y: Arr, n: int, Q: Any, layout: Layout) -> CoreResult:
    """Run bisection on the objective with the given layout.

    :return: a :class:`~corr_solver.types.CoreResult`
    """
    val, x = layout.initial(Y, n)
    omega = BSearchAdaptor(Q, Ell(val, x))
    upper = np.linalg.norm(Y, "fro") ** 2
    t, num_iters = bsearch(omega, (0.0, upper))
    return CoreResult(layout.extract(omega.x_best), num_iters, t != upper)


def lsq_corr_core(Y: Arr, n: int, Q: Any) -> CoreResult:
    """Least-squares core: bisection on the objective value.

    :param Y: biased sample covariance matrix
    :param n: number of coefficients
    :param Q: feasibility oracle
    :return: a :class:`~corr_solver.types.CoreResult`
    """
    return bsearch_core(Y, n, Q, LSQBsearchLayout())


def lsq_corr_core2(Y: Arr, n: int, omega: Any) -> CoreResult:
    """Least-squares core: augmented ``(x, t)`` cutting-plane optimization.

    :param Y: biased sample covariance matrix
    :param n: number of coefficients
    :param omega: optimization oracle
    :return: a :class:`~corr_solver.types.CoreResult`
    """
    res = cutting_plane_core(Y, n, omega, LSQAugmentedLayout())
    if res.coeffs is None:
        return CoreResult(np.zeros(n), res.num_iters, False)
    return res


def mle_corr_core(Y: Arr, n: int, omega: Any) -> CoreResult:
    """Maximum-likelihood core.

    :param Y: biased sample covariance matrix (unused; kept for the core signature)
    :param n: number of coefficients
    :param omega: optimization oracle
    :return: a :class:`~corr_solver.types.CoreResult`
    """
    return cutting_plane_core(Y, n, omega, MleLayout())
