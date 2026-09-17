"""
B-spline correlation oracle.

Fits a smooth, monotonically decreasing B-spline curve to a biased covariance
matrix:

1. ``generate_bspline_info`` builds the knot vector ``t`` and evaluates the
   B-spline basis functions (``Sigma``) on the pairwise site distance matrix.
2. ``mono_oracle`` / ``mono_decreasing_oracle2`` enforce the monotonic
   non-increasing constraint on the B-spline coefficients.

3. ``corr_bspline`` drives the cutting-plane oracle and returns the fitted
   ``BSpline`` object, the iteration count, and a feasibility flag.










"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import BSpline

from .corr_oracle import construct_distance_matrix

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


def mono_oracle(x: Arr) -> Optional[Tuple[Arr, float]]:
    """
    The `mono_oracle` function is an oracle that checks if a given array `x` satisfies the monotonic
    decreasing constraint and returns the gradient and the first violation if it exists.

    :param x: The parameter `x` is a list or array of numbers. It represents a sequence of values that
        we want to check for the monotonic decreasing constraint
    :return: The function `mono_oracle` returns two values: `g` and `fj`. `g` is a numpy array of zeros
        with the same length as `x`, where elements are set to -1.0 and 1.0 to enforce the monotonic
        decreasing constraint. `fj` is the difference between the next element and the current element in
        `x`.
    """
    # monotonic decreasing constraint
    n = len(x)
    g = np.zeros(n)
    for i in range(n - 1):
        if (fj := x[i + 1] - x[i]) > 0.0:
            g[i] = -1.0
            g[i + 1] = 1.0
            return g, fj
    return None


# The `mono_decreasing_oracle2` class is an oracle that checks if a given sequence is monotonically
# decreasing.
class mono_decreasing_oracle2:
    """Oracle for monotonic decreasing constraint.

    Wraps a basis oracle and enforces that the sequence of B-spline
    coefficients must be monotonically non-increasing (non-increasing x[i] >= x[i+1]).
    """

    def __init__(self, basis: Any) -> None:
        self.basis = basis

    def assess_optim(self, x: Arr, t: float) -> Tuple[Cut, Optional[float]]:
        """
        The function assess_optim assesses the optimality of a given solution by checking if it satisfies a
        monotonic decreasing constraint, and if not, it calls another function to assess optimality.

        :param x: An array of values
        :type x: Arr
        :param t: The parameter `t` represents the best-so-far optimal value. It is a float value that is
            used in the function to assess the optimality of a solution
        :type t: float
        :return: The function `assess_optim` returns a tuple containing a `Cut` object and an optional float
            value.
        """
        # monotonic decreasing constraint
        n = len(x)
        g = np.zeros(n)
        if cut := mono_oracle(x[:-1]):
            g1, fj = cut
            g[:-1] = g1
            g[-1] = 0.0
            return (g, fj), None
        return self.basis.assess_optim(x, t)


def corr_bspline(
    Y: Arr, site: Arr, m: int, oracle: Any, corr_core: Any
) -> Tuple[Any, int, bool]:
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
    Sigma, t, k = generate_bspline_info(site, m)
    Pb = oracle(Sigma, Y)
    omega = mono_decreasing_oracle2(Pb)
    c, num_iters, feasible = corr_core(Y, m, omega)
    return BSpline(t, c, k), num_iters, feasible


def generate_bspline_info(site: Arr, m: int) -> Tuple[List[Arr], np.ndarray, int]:
    """
    The function `generate_bspline_info` generates B-spline information given a set of points and a
    desired number of B-splines.

    :param site: The parameter `site` is a list or array of data points that define the shape or curve that
        you want to approximate using B-splines
    :param m: The parameter `m` represents the number of B-spline basis functions to generate. It
        determines the number of basis functions that will be used to approximate the input data
    :return: The function `generate_bspline_info` returns three values: `Sigma`, `t`, and `k`.
    """
    k = 2  # quadratic bspline
    h = site[-1] - site[0]
    d = np.sqrt(h @ h)
    t = np.linspace(0, d * 1.2, m + k + 1)
    spls = []
    for i in range(m):
        coeff = np.zeros(m)
        coeff[i] = 1
        spls += [BSpline(t, coeff, k)]
    D = construct_distance_matrix(site)
    Sigma = []
    for i in range(m):
        Sigma += [spls[i](D)]
    return Sigma, t, k
