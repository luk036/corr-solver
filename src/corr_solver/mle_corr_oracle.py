"""
Shared base class for the maximum-likelihood estimation oracles.

:class:`MleOptimOracle` provides the ``assess_optim`` template shared by the
MLE-style oracles: check feasibility, take a positive-semidefinite square root
of the feasible matrix, and compare the objective and its gradient against the
best-so-far value ``t``. The concrete MLE solver is the convex-concave procedure
in :mod:`corr_solver.cccp_mle_oracle`.
"""

from typing import Any, Optional, Tuple

from .types import Arr, Assessment, Cut


class MleOptimOracle:
    """Template method for the MLE-style ``assess_optim`` control flow.

    The shared skeleton checks feasibility, takes a PSD square root of the
    feasible matrix, and compares the objective against the best-so-far value
    ``t``. Subclasses plug in the pieces through :meth:`_first_cut`,
    :meth:`_sqrt` and :meth:`_value_and_grad`.
    """

    lmi0: Any

    def _first_cut(self, x: Arr) -> Optional[Cut]:
        """Return a feasibility cut to short-circuit, else ``None``."""
        return None

    def _sqrt(self) -> Arr:
        """Return upper-triangular ``R`` with the feasible matrix ``= R^T R``."""
        raise NotImplementedError

    def _value_and_grad(self, x: Arr, R: Arr) -> Tuple[float, Arr]:
        """Return the objective value and its gradient for factor ``R``."""
        raise NotImplementedError

    def assess_optim(self, x: Arr, t: float) -> Assessment:
        """Assess feasibility then optimality against best-so-far ``t``.

        :param x: coefficient vector
        :param t: best-so-far optimal value
        :return: an :class:`~corr_solver.types.Assessment`
        """
        if cut := self._first_cut(x):
            return Assessment(cut, None)

        R = self._sqrt()
        f1, g = self._value_and_grad(x, R)

        if (f := f1 - t) >= 0:
            return Assessment((g, f), None)
        return Assessment((g, 0.0), f1)
