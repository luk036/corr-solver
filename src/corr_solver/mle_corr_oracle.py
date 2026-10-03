"""
Maximum-likelihood estimation oracle.

Fits coefficients ``p`` of ``Omega(p)`` to a biased sample covariance matrix
``Y`` by minimizing ``log det Omega(p) + Tr(Omega(p)^{-1} Y)`` subject to
``2Y >= Omega(p) >= 0``. Feasibility is checked with the ``lmi`` and ``lmi0``
oracles; the objective and its gradient are then compared against the
best-so-far value ``t``.
"""

from typing import Any, List, Optional, Tuple

from .backends import default_backend
from .math_utils import mle_value_and_grad
from .protocols import OracleBackend
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


class mle_oracle(MleOptimOracle):
    """Maximum likelihood estimation:

    min  log det Ω(p) + Tr( Ω(p)^{-1} Y )
    s.t. 2Y ⪰ Ω(p) ⪰ 0,
    """

    def __init__(
        self, Sigma: List[Arr], Y: Arr, backend: Optional[OracleBackend] = None
    ):
        if backend is None:
            backend = default_backend()
        self.Y = Y
        self.Sigma = Sigma
        self.lmi0 = backend.lmi0(Sigma)
        self.lmi = backend.lmi(Sigma, 2 * Y)

    def _first_cut(self, x: Arr) -> Optional[Cut]:
        if cut := self.lmi.assess_feas(x):
            return cut
        return self.lmi0.assess_feas(x)

    def _sqrt(self) -> Arr:
        return self.lmi0.sqrt()

    def _value_and_grad(self, x: Arr, R: Arr) -> Tuple[float, Arr]:
        return mle_value_and_grad(R, self.Y, self.Sigma)


MLEOracle = mle_oracle
