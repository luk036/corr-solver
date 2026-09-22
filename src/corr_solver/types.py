"""Shared type aliases and result objects for corr-solver."""

from typing import Any, NamedTuple, Tuple

import numpy as np

Arr = np.ndarray
"""A NumPy array."""

Cut = Tuple[Arr, float]
"""A separating cut: ``(subgradient, violation)``."""


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
