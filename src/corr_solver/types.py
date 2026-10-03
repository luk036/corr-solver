"""Shared type aliases and result objects for corr-solver."""

from typing import Any, NamedTuple, Optional, Tuple

import numpy as np

Arr = np.ndarray
"""A NumPy array."""

Cut = Tuple[Arr, float]
"""A separating cut: ``(subgradient, violation)``."""


class Assessment(NamedTuple):
    """An optimization oracle's verdict.

    ``cut`` is the separating hyperplane and ``best`` is the improved
    best-so-far value when one was found, else ``None``. A :class:`NamedTuple`,
    so it still unpacks as the historical ``(cut, value)`` pair that the
    ``ellalgo`` drivers expect (``cut, gamma1 = oracle.assess_optim(...)``).
    """

    cut: Cut
    best: Optional[float]


class CoreResult(NamedTuple):
    """Result of a solver core.

    :param coeffs: fitted coefficients, or ``None`` when the core found none
    :param num_iters: number of cutting-plane / bisection iterations
    :param feasible: whether a feasible solution was found
    """

    coeffs: Optional[Arr]
    num_iters: int
    feasible: bool


class FitResult(NamedTuple):
    """Result of a correlation fit.

    A :class:`~typing.NamedTuple`, so it still unpacks as the historical
    ``(curve, num_iters, feasible)`` triple.

    :param curve: the fitted curve (``numpy.poly1d`` or a SciPy ``BSpline``)
    :param num_iters: number of cutting-plane iterations
    :param feasible: whether a feasible solution was found
    """

    curve: Any
    num_iters: int
    feasible: bool
