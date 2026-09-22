"""Structural interfaces shared by the correlation oracles.

These are :class:`typing.Protocol` definitions, so the pure-Python oracles, the
Numba drop-ins and the ``ellalgo`` oracles all satisfy them structurally without
inheriting from anything.
"""

from typing import Any, List, Optional, Protocol, Tuple

from .types import Arr, Cut


class FeasibilityOracle(Protocol):
    """Separation oracle: return a cut when ``x`` is infeasible, else ``None``."""

    def assess_feas(self, x: Arr) -> Optional[Cut]:
        """Assess feasibility of ``x``."""
        ...


class OptimizationOracle(Protocol):
    """Optimization oracle: assess ``x`` against the best-so-far value ``t``."""

    def assess_optim(self, x: Arr, t: float) -> Tuple[Cut, Optional[float]]:
        """Return a ``(cut, value)`` pair."""
        ...


class MatrixFunction(Protocol):
    """A symmetric matrix-valued function ``H(x)``, as consumed by ``GMIOracle``."""

    def eval(self, row: int, col: int, x: Arr) -> float:
        """Evaluate the ``(row, col)`` entry of ``H(x)``."""
        ...

    def neg_grad_sym_quad(self, Q: Any, x: Arr) -> Arr:
        """Return the negative gradient over the block described by ``Q``."""
        ...


class OracleBackend(Protocol):
    """Factory for a whole family of oracles (pure Python or Numba)."""

    def lmi0(self, F: List[Arr]) -> FeasibilityOracle:
        """Return an oracle for ``sum_k F_k x_k >= 0``."""
        ...

    def lmi(self, F: List[Arr], F0: Arr) -> FeasibilityOracle:
        """Return an oracle for ``F0 - sum_k F_k x_k >= 0``."""
        ...

    def qmi(self, F: List[Arr], F0: Arr) -> Any:
        """Return a quadratic-matrix-inequality oracle."""
        ...

    def lsq(self, F: List[Arr], F0: Arr) -> OptimizationOracle:
        """Return a least-squares optimization oracle."""
        ...

    def mle(self, Sigma: List[Arr], Y: Arr) -> OptimizationOracle:
        """Return a maximum-likelihood optimization oracle."""
        ...
