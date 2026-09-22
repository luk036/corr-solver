"""
Basis strategies and the unified correlation-fit driver.

A :class:`Basis` turns a site layout and a degree into the matrix basis consumed
by the cutting-plane oracles, and converts the resulting coefficients back into a
callable curve:

1. ``construct_distance_matrix`` / ``construct_poly_matrix`` build the distance
   and polynomial-basis matrices.
2. ``PolynomialBasis`` and ``BSplineBasis`` are the two :class:`Basis` strategies;
   the B-spline strategy also wraps the oracle with ``mono_decreasing_oracle2``.
3. ``fit`` is the shared driver behind ``corr_poly`` and ``corr_bspline``.
"""

from typing import Any, Callable, List, Optional, Protocol, Tuple

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.distance import pdist, squareform

from .types import Arr, Cut, FitResult


def construct_distance_matrix(site: Arr) -> Arr:
    """
    The function `construct_distance_matrix` takes in a list of site locations and returns a distance
    matrix object where each element represents the distance between two sites.

    :param site: The parameter `site` is the location of sites. It is an array that contains the coordinates
        of each site
    :type site: Arr
    :return: a distance matrix object.
    """
    return squareform(pdist(site))


def construct_poly_matrix(site: Arr, m: int) -> List[Arr]:
    """
    The function `construct_poly_matrix` takes in a list of site locations `site` and a degree `m`, and
    returns a list of distance matrices for a polynomial of degree `m`.

    :param site: The parameter `site` is the location of sites, which is expected to be an array. It
        represents the locations of the sites for which the distance matrix is being constructed
    :type site: Arr
    :param m: The parameter `m` represents the degree of the polynomial. It determines the number of
        distance matrices that will be constructed
    :return: The function `construct_poly_matrix` returns a list of arrays.
    """
    n = len(site)
    D1 = construct_distance_matrix(site)
    D = np.ones((n, n))
    Sigma = [D]
    for _ in range(m - 1):
        D = np.multiply(D, D1)
        Sigma += [D]
    return Sigma


def mono_oracle(x: Arr) -> Optional[Cut]:
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

    :param basis: the wrapped oracle
    :param n_coeff: number of leading entries of ``x`` that are control
        coefficients; defaults to ``len(x) - 1`` (drops a trailing objective
        variable). Pass ``m`` explicitly when the solver core passes coefficients
        only, as the MLE core does.
    """

    def __init__(self, basis: Any, n_coeff: Optional[int] = None) -> None:
        self.basis = basis
        self.n_coeff = n_coeff

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
        n = len(x)
        k = n - 1 if self.n_coeff is None else self.n_coeff
        g = np.zeros(n)
        if cut := mono_oracle(x[:k]):
            g1, fj = cut
            g[:k] = g1
            return (g, fj), None
        return self.basis.assess_optim(x, t)


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
    if m < k + 1:
        raise ValueError(
            f"quadratic B-spline needs m >= {k + 1} control points, got {m}"
        )
    D = construct_distance_matrix(site)
    dmax = float(D.max())
    interior = np.linspace(0.0, dmax, m - k - 1 + 2)[1:-1]
    t = np.concatenate((np.zeros(k + 1), interior, np.full(k + 1, dmax)))
    spls = []
    for i in range(m):
        coeff = np.zeros(m)
        coeff[i] = 1
        spls += [BSpline(t, coeff, k)]
    Sigma = [spls[i](D) for i in range(m)]
    return Sigma, t, k


class Basis(Protocol):
    """Strategy that builds a basis and packages the fitted coefficients."""

    def build(self, site: Arr, m: int) -> List[Arr]:
        """Return the ``m`` basis matrices for the given sites."""
        ...

    def wrap(self, oracle: Any, m: int) -> Any:
        """Wrap the basis oracle with any extra constraint."""
        ...

    def curve(self, coeffs: Arr) -> Callable[[Arr], Arr]:
        """Turn coefficients into a callable curve."""
        ...


class PolynomialBasis:
    """Power-series basis (``Sigma_k = D.^k``); the curve is a ``numpy.poly1d``."""

    def build(self, site: Arr, m: int) -> List[Arr]:
        """Return the polynomial basis matrices."""
        return construct_poly_matrix(site, m)

    def wrap(self, oracle: Any, m: int) -> Any:
        """No extra constraint for the polynomial basis."""
        return oracle

    def curve(self, coeffs: Arr) -> Callable[[Arr], Arr]:
        """Return the polynomial as a ``numpy.poly1d``."""
        return np.poly1d(np.ascontiguousarray(coeffs[::-1]))


class BSplineBasis:
    """Quadratic B-spline basis with a monotone-decreasing coefficient constraint."""

    def build(self, site: Arr, m: int) -> List[Arr]:
        """Return the B-spline basis matrices, keeping the knots for :meth:`curve`."""
        self.Sigma, self.t, self.k = generate_bspline_info(site, m)
        return self.Sigma

    def wrap(self, oracle: Any, m: int) -> Any:
        """Enforce monotonically non-increasing coefficients."""
        return mono_decreasing_oracle2(oracle, m)

    def curve(self, coeffs: Arr) -> Callable[[Arr], Arr]:
        """Return the B-spline."""
        return BSpline(self.t, coeffs, self.k)


def fit(
    Y: Arr,
    site: Arr,
    m: int,
    oracle: Any,
    corr_core: Any,
    basis: Optional[Basis] = None,
) -> FitResult:
    """Build the basis, wrap the oracle, run the core, and package the curve.

    :param Y: biased sample covariance matrix
    :param site: site locations
    :param m: degree / number of basis functions
    :param oracle: oracle factory ``(Sigma, Y) -> oracle``
    :param corr_core: cutting-plane solver core
    :param basis: basis strategy; defaults to :class:`PolynomialBasis`
    :return: the fitted curve, the iteration count and a feasibility flag
    """
    basis = PolynomialBasis() if basis is None else basis
    Sigma = basis.build(site, m)
    Pb = oracle(Sigma, Y)
    omega = basis.wrap(Pb, m)
    c, num_iters, feasible = corr_core(Y, m, omega)
    return FitResult(basis.curve(c), num_iters, feasible)
