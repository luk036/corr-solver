"""
B-spline correlation oracle.

Fits a smooth, monotonically decreasing B-spline curve to a biased covariance
matrix via the quadratic B-spline basis in :mod:`corr_solver.basis`.

``generate_bspline_info``, ``mono_oracle`` and ``mono_decreasing_oracle2`` live
in :mod:`corr_solver.basis` and are re-exported here for backward compatibility.
"""

from typing import Any

from .basis import (
    BSplineBasis,
    fit,
    generate_bspline_info,
    mono_decreasing_oracle2,
    mono_oracle,
)
from .types import Arr, FitResult

__all__ = [
    "corr_bspline",
    "generate_bspline_info",
    "mono_oracle",
    "mono_decreasing_oracle2",
]


def corr_bspline(Y: Arr, site: Arr, m: int, oracle: Any, corr_core: Any) -> FitResult:
    """
    The `corr_bspline` function takes in input parameters `Y`, `site`, `m`, `oracle`, and `corr_core`, and
    returns a BSpline object, the number of iterations, and a feasibility indicator.

    :param Y: The input data Y for the B-spline algorithm
    :param site: The parameter `site` represents the number of control points in the B-spline curve. It
        determines the flexibility and smoothness of the curve
    :param m: The parameter `m` represents the number of control points in the B-spline curve. It
        determines the flexibility and smoothness of the curve. A higher value of `m` will result in a more
        flexible curve that can better fit the data, but it may also lead to overfitting
    :param oracle: The `oracle` parameter is a separation oracle
    :param corr_core: The `corr_core` parameter is a function that takes in the following arguments:
    :return: The function `corr_bspline` returns three values:
    """
    return fit(Y, site, m, oracle, corr_core, BSplineBasis())
