"""Oracle backends: the pure-Python and Numba families behind one factory.

``PurePythonBackend`` is always importable; ``NumbaBackend`` lives in
:mod:`corr_solver.numba_oracles` (which needs the optional ``numba`` dependency)
and is imported lazily by :func:`make_backend`.

Both families must return the same cuts, so the arithmetic order is part of the
contract: the quadratic matrix inequality is evaluated on the boundary of the
positive-semidefinite cone, where the first pivot to turn negative is decided by
rounding. Do not reorder the sums.
"""

from typing import Callable, Dict, List, Optional

from ellalgo.oracles.lmi0_oracle import LMI0Oracle
from ellalgo.oracles.lmi_oracle import LMIOracle

from .lsq_corr_oracle import lsq_oracle
from .mle_corr_oracle import mle_oracle
from .protocols import (
    FeasibilityOracle,
    OptimizationOracle,
    OracleBackend,
    QuadraticOracle,
    SqrtFeasibilityOracle,
)
from .qmi_oracle import QMIOracle
from .types import Arr, Cut


class PureLMI0Oracle:
    """``ellalgo``'s ``LMI0Oracle`` plus the uniform :meth:`sqrt` accessor.

    ``mle_oracle`` asks every backend's ``lmi0`` for ``sqrt``; ``ellalgo`` only
    exposes it through ``ldlt_mgr``, so this adapter levels the interface.
    Composing rather than subclassing keeps it decoupled from ellalgo internals.
    """

    def __init__(self, F: List[Arr]) -> None:
        self._inner = LMI0Oracle(F)

    def assess_feas(self, x: Arr) -> Optional[Cut]:
        """Return a cut when ``sum_k F_k x_k`` is not positive semidefinite."""
        return self._inner.assess_feas(x)

    def sqrt(self) -> Arr:
        """Return upper-triangular ``R`` with ``sum_k F_k x_k = R^T R``."""
        return self._inner.ldlt_mgr.sqrt()


class PurePythonBackend:
    """Oracles written in pure Python (``ellalgo`` plus this package)."""

    def lmi0(self, F: List[Arr]) -> SqrtFeasibilityOracle:
        """Return an oracle for ``sum_k F_k x_k >= 0``."""
        return PureLMI0Oracle(F)

    def lmi(self, F: List[Arr], F0: Arr) -> FeasibilityOracle:
        """Return an oracle for ``F0 - sum_k F_k x_k >= 0``."""
        return LMIOracle(F, F0)

    def qmi(self, F: List[Arr], F0: Arr) -> QuadraticOracle:
        """Return a quadratic-matrix-inequality oracle."""
        return QMIOracle(F, F0)

    def lsq(self, F: List[Arr], F0: Arr) -> OptimizationOracle:
        """Return a least-squares optimization oracle."""
        return lsq_oracle(F, F0, self)

    def mle(self, Sigma: List[Arr], Y: Arr) -> OptimizationOracle:
        """Return a maximum-likelihood optimization oracle."""
        return mle_oracle(Sigma, Y, self)


def _numba_backend() -> OracleBackend:
    """Import and construct the Numba backend lazily (optional dependency)."""
    from .numba_oracles import NumbaBackend

    return NumbaBackend()


_BACKENDS: Dict[str, Callable[[], OracleBackend]] = {
    "python": PurePythonBackend,
    "numba": _numba_backend,
}


def make_backend(name: str = "python") -> OracleBackend:
    """Return an oracle backend by name.

    :param name: ``"python"`` (default) or ``"numba"``
    :return: the matching backend
    """
    try:
        factory = _BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown oracle backend: {name!r}") from None
    return factory()
