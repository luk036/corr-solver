"""
Basis strategies and the unified correlation-fit driver.

A :class:`Basis` turns a site layout and a degree into a :class:`BasisModel`,
which carries the matrix basis and the two operations the driver needs:

1. ``construct_distance_matrix`` / ``construct_poly_matrix`` build the distance
   and polynomial-basis matrices.
2. ``PolynomialBasis`` and ``BSplineBasis`` are the two :class:`Basis` factories;
   the B-spline model also wraps the oracle with ``MonotoneDecreasingOracle``.
3. ``fit`` is the shared driver behind ``corr_poly`` and ``corr_bspline``.
"""

from dataclasses import dataclass
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


# The `MonotoneDecreasingOracle` class is an oracle that checks if a given sequence is monotonically
# decreasing.
class MonotoneDecreasingOracle:
    """Oracle for the monotonic-decreasing constraint.

    Wraps another oracle and prepends a cut whenever the leading control
    coefficients are not monotonically non-increasing (``x[i] >= x[i+1]``). It
    forwards the feasibility, optimization and update interfaces, so it composes
    with the bisection cores as well as the cutting-plane ones; the wrapped
    oracle must provide whichever of ``assess_feas``/``update`` or
    ``assess_optim`` the chosen core drives.

    :param basis: the wrapped oracle
    :param n_coeff: number of leading entries of ``x`` that are control
        coefficients. ``None`` means "all of ``x``" for :meth:`assess_feas` and
        "``len(x) - 1``" for :meth:`assess_optim` (dropping a trailing objective
        variable). Pass ``m`` explicitly when the solver core passes coefficients
        only, as the MLE core does.
    """

    def __init__(self, basis: Any, n_coeff: Optional[int] = None) -> None:
        self.basis = basis
        self.n_coeff = n_coeff

    def _mono_cut(self, x: Arr, k: int) -> Optional[Cut]:
        g = np.zeros(len(x))
        if cut := mono_oracle(x[:k]):
            g1, fj = cut
            g[:k] = g1
            return g, fj
        return None

    def update(self, t: float) -> None:
        """Forward the best-so-far value to the wrapped feasibility oracle."""
        self.basis.update(t)

    def assess_feas(self, x: Arr) -> Optional[Cut]:
        """Return a monotonicity cut if ``x`` violates it, else delegate."""
        k = len(x) if self.n_coeff is None else self.n_coeff
        if (cut := self._mono_cut(x, k)) is not None:
            return cut
        return self.basis.assess_feas(x)

    def assess_optim(self, x: Arr, t: float) -> Tuple[Cut, Optional[float]]:
        """Return a monotonicity cut if ``x`` violates it, else delegate.

        :param x: An array of values
        :type x: Arr
        :param t: the best-so-far optimal value
        :type t: float
        :return: a ``(cut, value)`` pair
        """
        k = len(x) - 1 if self.n_coeff is None else self.n_coeff
        if (cut := self._mono_cut(x, k)) is not None:
            return cut, None
        return self.basis.assess_optim(x, t)


mono_decreasing_oracle2 = MonotoneDecreasingOracle
"""Backward-compatible alias for :class:`MonotoneDecreasingOracle`."""


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


@dataclass(frozen=True, eq=False)
class BasisModel:
    """A built basis: its matrices plus the oracle wrapper and curve factory."""

    matrices: List[Arr]
    wrap: Callable[[Any, int], Any]
    curve: Callable[[Arr], Any]


class Basis(Protocol):
    """Strategy that builds a :class:`BasisModel` for a site layout and degree."""

    def build(self, site: Arr, m: int) -> BasisModel:
        """Return the built basis for the given sites."""
        ...


class PolynomialBasis:
    """Power-series basis (``Sigma_k = D.^k``); the curve is a ``numpy.poly1d``."""

    def build(self, site: Arr, m: int) -> BasisModel:
        """Return the polynomial basis with no extra oracle constraint."""
        return BasisModel(
            matrices=construct_poly_matrix(site, m),
            wrap=lambda oracle, m: oracle,
            curve=lambda coeffs: np.poly1d(np.ascontiguousarray(coeffs[::-1])),
        )


class BSplineBasis:
    """Quadratic B-spline basis with a monotone-decreasing coefficient constraint."""

    def build(self, site: Arr, m: int) -> BasisModel:
        """Return the B-spline basis, capturing its knots inside the model."""
        matrices, t, k = generate_bspline_info(site, m)
        return BasisModel(
            matrices=matrices,
            wrap=lambda oracle, m: MonotoneDecreasingOracle(oracle, m),
            curve=lambda coeffs: BSpline(t, coeffs, k),
        )


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
    model = basis.build(site, m)
    Pb = oracle(model.matrices, Y)
    omega = model.wrap(Pb, m)
    c, num_iters, feasible = corr_core(Y, m, omega)
    return FitResult(model.curve(c), num_iters, feasible)
