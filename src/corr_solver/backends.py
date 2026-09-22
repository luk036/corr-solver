"""Oracle backends: the pure-Python and Numba families behind one factory.

``PurePythonBackend`` is always importable; ``NumbaBackend`` lives in
:mod:`corr_solver.numba_oracles` (which needs the optional ``numba`` dependency)
and is imported lazily by :func:`make_backend`.
"""

from typing import Any, List

from ellalgo.oracles.lmi0_oracle import LMI0Oracle
from ellalgo.oracles.lmi_oracle import LMIOracle

from .lsq_corr_oracle import lsq_oracle
from .mle_corr_oracle import mle_oracle
from .protocols import OracleBackend
from .qmi_oracle import QMIOracle
from .types import Arr


class PurePythonBackend:
    """Oracles written in pure Python (``ellalgo`` plus this package)."""

    def lmi0(self, F: List[Arr]) -> Any:
        """Return an oracle for ``sum_k F_k x_k >= 0``."""
        return LMI0Oracle(F)

    def lmi(self, F: List[Arr], F0: Arr) -> Any:
        """Return an oracle for ``F0 - sum_k F_k x_k >= 0``."""
        return LMIOracle(F, F0)

    def qmi(self, F: List[Arr], F0: Arr) -> Any:
        """Return a quadratic-matrix-inequality oracle."""
        return QMIOracle(F, F0)

    def lsq(self, F: List[Arr], F0: Arr) -> Any:
        """Return a least-squares optimization oracle."""
        return lsq_oracle(F, F0)

    def mle(self, Sigma: List[Arr], Y: Arr) -> Any:
        """Return a maximum-likelihood optimization oracle."""
        return mle_oracle(Sigma, Y)


def make_backend(name: str = "python") -> OracleBackend:
    """Return an oracle backend by name.

    :param name: ``"python"`` (default) or ``"numba"``
    :return: the matching backend
    """
    if name == "python":
        return PurePythonBackend()
    if name == "numba":
        from .numba_oracles import NumbaBackend

        return NumbaBackend()
    raise ValueError(f"unknown oracle backend: {name!r}")
