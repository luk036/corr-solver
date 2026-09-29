"""
JIT-compiled separation oracles (optional accelerator).

Drop-in replacements for :class:`~corr_solver.qmi_oracle.QMIOracle`,
:class:`~corr_solver.lsq_corr_oracle.lsq_oracle` and
:class:`~corr_solver.mle_corr_oracle.mle_oracle`, with the LDL^T factorization,
the basis-matrix assembly and the subgradient evaluation compiled by Numba.

The pure-Python oracles reach the same answer but spend most of their time in
the interpreter: the LDL^T factorization requests one matrix element at a time
through a Python callback, and each callback runs a Python-level ``sum`` over
the basis matrices. For the 20x20 problems used here that is roughly 120
interpreter calls per factorization. Compiling the kernels removes that
overhead.

Requires the optional ``numba`` dependency::

    pip install corr-solver[numba]

The kernels below are deliberate transcriptions of ``LDLTMgr._factor_impl``,
``LDLTMgr.witness``, ``LDLTMgr.sym_quad`` and the oracle ``eval`` methods,
keeping the same arithmetic order. That matters: the quadratic matrix
inequality is evaluated at the boundary of the positive-semidefinite cone,
where the first pivot to turn negative is decided by rounding, so a different
summation order picks a different separating cut and a different iteration
count. Do not "simplify" the loops into BLAS calls without re-checking the
iteration counts of the solver tests.
"""

from typing import List, Optional, Tuple

import numpy as np
from numba import njit

from .lsq_corr_oracle import lsq_oracle
from .mle_corr_oracle import mle_oracle
from .protocols import (
    FeasibilityOracle,
    OptimizationOracle,
    QuadraticOracle,
    SqrtFeasibilityOracle,
)
from .types import Arr, Cut


@njit(cache=True, fastmath=False)
def _ldlt(M: Arr, n: int, L: Arr, D: Arr, wit: Arr) -> int:  # pragma: no cover
    """LDL^T factorization mirroring ``LDLTMgr.factor``.

    Writes the unit lower-triangular factor into ``L``, the diagonal into ``D``
    and, when the matrix is not positive definite, the witness vector into
    ``wit``. Returns the position of the first non-positive pivot (0 if the
    matrix is positive definite).
    """
    start = 0
    pos1 = 0
    for i in range(n):
        diag = M[i, start]
        for j in range(start, i):
            L[j, i] = diag
            L[i, j] = diag / L[j, j]
            stop = j + 1
            acc = 0.0
            for k in range(start, stop):
                acc += L[i, k] * L[k, stop]
            diag = M[i, stop] - acc
        L[i, i] = diag
        D[i] = diag
        if diag < 0.0:
            pos1 = i + 1
            break
        if diag == 0.0:
            pos1 = i + 1
            break
    if pos1 != 0:
        m = pos1 - 1
        wit[m] = 1.0
        for i in range(m, start, -1):
            acc = 0.0
            for k in range(i, pos1):
                acc += L[k, i - 1] * wit[k]
            wit[i - 1] = -acc
    return pos1


@njit(cache=True, fastmath=False)
def _qmi_fx(Fstack: Arr, F0: Arr, x: Arr, Fx: Arr) -> None:  # pragma: no cover
    """Assemble ``Fx[r, c] = F0[c, r] - sum_k Fstack[k, c, r] * x[k]``."""
    ncol = F0.shape[0]
    mdim = F0.shape[1]
    nk = Fstack.shape[0]
    for r in range(mdim):
        for c in range(ncol):
            s = F0[c, r]
            for k in range(nk):
                s -= Fstack[k, c, r] * x[k]
            Fx[r, c] = s


@njit(cache=True, fastmath=False)
def _qmi_matrix(Fx: Arr, t: float, M: Arr) -> None:  # pragma: no cover
    """Assemble the quadratic matrix inequality ``M = t*I - Fx @ Fx.T``."""
    mdim = M.shape[0]
    ncol = Fx.shape[1]
    for i in range(mdim):
        for j in range(mdim):
            s = 0.0
            for c in range(ncol):
                s += Fx[i, c] * Fx[j, c]
            Mij = -s
            if i == j:
                Mij += t
            M[i, j] = Mij


@njit(cache=True, fastmath=False)
def _qmi_grad(
    Fstack: Arr, Fx: Arr, wit: Arr, pos1: int, g: Arr
) -> None:  # pragma: no cover
    """Assemble the subgradient from the witness vector over the failed block."""
    nk = Fstack.shape[0]
    ncol = Fx.shape[1]
    for k in range(nk):
        acc = 0.0
        for j in range(ncol):
            u = 0.0
            av = 0.0
            for i in range(pos1):
                u += wit[i] * Fstack[k, i, j]
                av += wit[i] * Fx[i, j]
            acc += u * av
        g[k] = -2.0 * acc


@njit(cache=True, fastmath=False)
def _qmi_assess(  # pragma: no cover
    Fstack: Arr,
    F0: Arr,
    x: Arr,
    t: float,
    Fx: Arr,
    M: Arr,
    L: Arr,
    D: Arr,
    wit: Arr,
    g: Arr,
) -> Tuple[bool, int]:
    """Feasibility check for ``t*I - F(x)^T F(x) >= 0`` with ``F(x) = F0 - sum_k F_k x_k``."""
    _qmi_fx(Fstack, F0, x, Fx)
    _qmi_matrix(Fx, t, M)
    pos1 = _ldlt(M, F0.shape[1], L, D, wit)
    if pos1 == 0:
        return True, 0
    _qmi_grad(Fstack, Fx, wit, pos1, g)
    return False, pos1


@njit(cache=True, fastmath=False)
def _lmi0_assess(  # pragma: no cover
    Fstack: Arr, x: Arr, M: Arr, L: Arr, D: Arr, wit: Arr, g: Arr
) -> Tuple[bool, float]:
    """Feasibility check for ``sum_k F_k x_k >= 0``."""
    ndim = Fstack.shape[1]
    nk = Fstack.shape[0]
    for i in range(ndim):
        for j in range(ndim):
            s = 0.0
            for k in range(nk):
                s += Fstack[k, i, j] * x[k]
            M[i, j] = s
    pos1 = _ldlt(M, ndim, L, D, wit)
    if pos1 == 0:
        return True, 0.0
    for k in range(nk):
        acc = 0.0
        for i in range(pos1):
            for j in range(pos1):
                acc += wit[i] * Fstack[k, i, j] * wit[j]
        g[k] = -acc
    return False, -D[pos1 - 1]


@njit(cache=True, fastmath=False)
def _lmi_assess(  # pragma: no cover
    Fstack: Arr, F0: Arr, x: Arr, M: Arr, L: Arr, D: Arr, wit: Arr, g: Arr
) -> Tuple[bool, float]:
    """Feasibility check for ``F0 - sum_k F_k x_k >= 0``."""
    ndim = F0.shape[0]
    nk = Fstack.shape[0]
    for i in range(ndim):
        for j in range(ndim):
            s = 0.0
            for k in range(nk):
                s += Fstack[k, i, j] * x[k]
            M[i, j] = F0[i, j] - s
    pos1 = _ldlt(M, ndim, L, D, wit)
    if pos1 == 0:
        return True, 0.0
    for k in range(nk):
        acc = 0.0
        for i in range(pos1):
            for j in range(pos1):
                acc += wit[i] * Fstack[k, i, j] * wit[j]
        g[k] = acc
    return False, -D[pos1 - 1]


class NumbaQMIOracle:
    """Quadratic matrix inequality oracle, JIT-compiled.

    Drop-in replacement for :class:`~corr_solver.qmi_oracle.QMIOracle`.

    :param F: feature matrices, each of shape ``(n, m)``
    :param F0: reference distribution matrix of shape ``(n, m)``
    """

    def __init__(self, F: List[Arr], F0: Arr) -> None:
        self.Fstack = np.ascontiguousarray(F, dtype=np.float64)
        self.F0 = np.ascontiguousarray(F0, dtype=np.float64)
        self.Fx = np.zeros_like(self.F0.T)
        mdim = F0.shape[1]
        self.M = np.zeros((mdim, mdim))
        self.L = np.zeros((mdim, mdim))
        self.D = np.zeros(mdim)
        self.wit = np.zeros(mdim)
        self.g = np.zeros(len(F))
        self.t = 0.0
        self.pos1 = 0

    def update(self, t: float) -> None:
        """Set the best-so-far objective value ``t``.

        :param t: best-so-far optimal value
        """
        self.t = t

    def assess_feas(self, x: Arr) -> Optional[Cut]:
        """Assess feasibility of ``x``.

        :param x: candidate point
        :return: a separating cut ``(g, ep)`` if infeasible, otherwise ``None``
        """
        ok, pos1 = _qmi_assess(
            self.Fstack,
            self.F0,
            x,
            self.t,
            self.Fx,
            self.M,
            self.L,
            self.D,
            self.wit,
            self.g,
        )
        self.pos1 = pos1
        if ok:
            return None
        return self.g.copy(), -self.D[pos1 - 1]

    def witness_sq(self) -> float:
        """Return ``wit @ wit`` over the failed block, mirroring the pure oracle."""
        wit = self.wit[: self.pos1]
        return float(wit @ wit)


class NumbaLMI0Oracle:
    """Oracle for ``sum_k F_k x_k >= 0``, JIT-compiled.

    Drop-in replacement for ``ellalgo.oracles.lmi0_oracle.LMI0Oracle``.

    :param mat_f: list of symmetric coefficient matrices ``[F_1, ..., F_n]``
    """

    def __init__(self, mat_f: List[Arr]) -> None:
        self.Fstack = np.ascontiguousarray(mat_f, dtype=np.float64)
        ndim = mat_f[0].shape[0]
        self.M = np.zeros((ndim, ndim))
        self.L = np.zeros((ndim, ndim))
        self.D = np.zeros(ndim)
        self.wit = np.zeros(ndim)
        self.g = np.zeros(len(mat_f))

    def assess_feas(self, x: Arr) -> Optional[Cut]:
        """Assess feasibility of ``x``.

        :param x: candidate point
        :return: a separating cut ``(g, ep)`` if infeasible, otherwise ``None``
        """
        ok, ep = _lmi0_assess(self.Fstack, x, self.M, self.L, self.D, self.wit, self.g)
        if ok:
            return None
        return self.g.copy(), ep

    def sqrt(self) -> Arr:
        """Return the upper-triangular ``R`` with ``F(x) = R^T R``.

        :return: upper triangular Cholesky factor
        """
        ndim = self.D.shape[0]
        R = np.zeros((ndim, ndim))
        for i in range(ndim):
            R[i, i] = np.sqrt(self.D[i])
            for j in range(i + 1, ndim):
                R[i, j] = self.L[j, i] * R[i, i]
        return R


class NumbaLMIOracle:
    """Oracle for ``F0 - sum_k F_k x_k >= 0``, JIT-compiled.

    Drop-in replacement for ``ellalgo.oracles.lmi_oracle.LMIOracle``.

    :param mat_f: list of symmetric coefficient matrices ``[F_1, ..., F_n]``
    :param mat_b: constant matrix ``F0``
    """

    def __init__(self, mat_f: List[Arr], mat_b: Arr) -> None:
        self.Fstack = np.ascontiguousarray(mat_f, dtype=np.float64)
        self.F0 = np.ascontiguousarray(mat_b, dtype=np.float64)
        ndim = mat_b.shape[0]
        self.M = np.zeros((ndim, ndim))
        self.L = np.zeros((ndim, ndim))
        self.D = np.zeros(ndim)
        self.wit = np.zeros(ndim)
        self.g = np.zeros(len(mat_f))

    def assess_feas(self, x: Arr) -> Optional[Cut]:
        """Assess feasibility of ``x``.

        :param x: candidate point
        :return: a separating cut ``(g, ep)`` if infeasible, otherwise ``None``
        """
        ok, ep = _lmi_assess(
            self.Fstack, self.F0, x, self.M, self.L, self.D, self.wit, self.g
        )
        if ok:
            return None
        return self.g.copy(), ep


class NumbaLsqOracle(lsq_oracle):
    """Least-squares oracle bound to the Numba leaves; the logic lives in the base.

    :param F: basis matrices ``[F_1, ..., F_n]``
    :param F0: reference matrix
    """

    def __init__(self, F: List[Arr], F0: Arr) -> None:
        super().__init__(F, F0, NumbaBackend())


class NumbaMleOracle(mle_oracle):
    """Maximum-likelihood oracle bound to the Numba leaves; logic lives in the base.

    :param Sigma: basis matrices ``[Sigma_1, ..., Sigma_n]``
    :param Y: biased sample covariance matrix
    """

    def __init__(self, Sigma: List[Arr], Y: Arr) -> None:
        super().__init__(Sigma, Y, NumbaBackend())


class NumbaBackend:
    """Numba-compiled oracle family, mirroring ``PurePythonBackend``."""

    def lmi0(self, mat_f: List[Arr]) -> SqrtFeasibilityOracle:
        """Return an oracle for ``sum_k F_k x_k >= 0``."""
        return NumbaLMI0Oracle(mat_f)

    def lmi(self, mat_f: List[Arr], mat_b: Arr) -> FeasibilityOracle:
        """Return an oracle for ``F0 - sum_k F_k x_k >= 0``."""
        return NumbaLMIOracle(mat_f, mat_b)

    def qmi(self, F: List[Arr], F0: Arr) -> QuadraticOracle:
        """Return a quadratic-matrix-inequality oracle."""
        return NumbaQMIOracle(F, F0)

    def lsq(self, F: List[Arr], F0: Arr) -> OptimizationOracle:
        """Return a least-squares optimization oracle."""
        return NumbaLsqOracle(F, F0)

    def mle(self, Sigma: List[Arr], Y: Arr) -> OptimizationOracle:
        """Return a maximum-likelihood optimization oracle."""
        return NumbaMleOracle(Sigma, Y)
