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

from typing import Any, List, Optional, Tuple

import numpy as np
from numba import njit

Arr = np.ndarray
Cut = Tuple[Arr, float]


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
    ncol = F0.shape[0]
    mdim = F0.shape[1]
    nk = Fstack.shape[0]
    for r in range(mdim):
        for c in range(ncol):
            s = F0[c, r]
            for k in range(nk):
                s -= Fstack[k, c, r] * x[k]
            Fx[r, c] = s
    for i in range(mdim):
        for j in range(mdim):
            s = 0.0
            for c in range(ncol):
                s += Fx[i, c] * Fx[j, c]
            Mij = -s
            if i == j:
                Mij += t
            M[i, j] = Mij
    pos1 = _ldlt(M, mdim, L, D, wit)
    if pos1 == 0:
        return True, 0
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


class NumbaLsqOracle:
    """Least-squares correlation oracle, JIT-compiled.

    Drop-in replacement for :class:`~corr_solver.lsq_corr_oracle.lsq_oracle`.

    :param F: basis matrices ``[F_1, ..., F_n]``
    :param F0: reference matrix
    """

    def __init__(self, F: List[Arr], F0: Arr) -> None:
        self.qmi = NumbaQMIOracle(F, F0)
        self.lmi0 = NumbaLMI0Oracle(F)

    def assess_optim(self, x: Arr, t: float) -> Tuple[Cut, Optional[float]]:
        """Assess feasibility and optimality of ``x`` against best-so-far ``t``.

        :param x: candidate point, the last entry being the objective variable
        :param t: best-so-far optimal value
        :return: a ``(cut, value)`` pair, mirroring ``lsq_oracle.assess_optim``
        """
        n = len(x)
        g = np.zeros(n)
        cut = self.lmi0.assess_feas(x[:-1])
        if cut is not None:
            g[:-1] = cut[0]
            return (g, cut[1]), None
        self.qmi.update(x[-1])
        cut = self.qmi.assess_feas(x[:-1])
        if cut is not None:
            g[:-1] = cut[0]
            wit = self.qmi.wit[: self.qmi.pos1]
            g[-1] = -float(wit @ wit)
            return (g, cut[1]), None
        g[-1] = 1
        tc = x[-1]
        if (fj := tc - t) > 0.0:
            return (g, fj), None
        return (g, 0.0), tc


class NumbaMleOracle:
    """Maximum-likelihood estimation oracle, JIT-compiled.

    Drop-in replacement for :class:`~corr_solver.mle_corr_oracle.mle_oracle`.

    :param Sigma: basis matrices ``[Sigma_1, ..., Sigma_n]``
    :param Y: biased sample covariance matrix
    """

    def __init__(self, Sigma: List[Arr], Y: Arr) -> None:
        self.Y = Y
        self.Sigma = Sigma
        self.lmi0 = NumbaLMI0Oracle(Sigma)
        self.lmi = NumbaLMIOracle(Sigma, 2 * Y)

    def assess_optim(self, x: Arr, t: float) -> Tuple[Cut, Optional[float]]:
        """Assess feasibility and optimality of ``x`` against best-so-far ``t``.

        :param x: coefficient vector
        :param t: best-so-far optimal value
        :return: a ``(cut, value)`` pair, mirroring ``mle_oracle.assess_optim``
        """
        cut = self.lmi.assess_feas(x)
        if cut is not None:
            return cut, None
        cut = self.lmi0.assess_feas(x)
        if cut is not None:
            return cut, None
        R = self.lmi0.sqrt()
        invR = np.linalg.inv(R)
        S = invR @ invR.T
        SY = S @ self.Y
        f1 = 2 * np.sum(np.log(np.diag(R))) + np.trace(SY)
        g: Any = np.empty(len(x))
        V = S - SY @ S
        for i in range(len(x)):
            g[i] = np.sum(V.T * self.Sigma[i])
        if (f := f1 - t) >= 0:
            return (g, f), None
        return (g, 0.0), f1
