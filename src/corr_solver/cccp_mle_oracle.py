"""Convex-concave procedure (CCP) for the family-constrained MLE.

The objective ``f(Omega) = log det Omega + Tr(Omega^-1 Y)`` is a difference of
convex functions: ``Tr(Omega^-1 Y)`` is convex in ``Omega`` and ``-log det
Omega`` is convex. Each round linearizes the concave part at the current
iterate ``Omega_k``, giving the convex surrogate

    Tr(Omega^-1 Y) + Tr(Omega_k^-1 Omega) + const,

which is minimized over ``Omega >= 0`` with the same cutting-plane machinery.
The surrogate carries no ``Omega <= 2Y`` upper bound, so it removes the
non-convexity that the ``2Y`` constraint of
:class:`~corr_solver.mle_corr_oracle.mle_oracle` was papering over.
"""

from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from ellalgo.cutting_plane import cutting_plane_optim
from ellalgo.ell import Ell
from ellalgo.oracles.lmi0_oracle import LMI0Oracle

Arr = np.ndarray
Cut = Tuple[Arr, float]


class cccp_mle_oracle:
    """Oracle for a single CCP round of the MLE.

    :param Sigma: basis matrices ``[Sigma_1, ..., Sigma_n]``
    :param Y: biased sample covariance matrix
    :param M: ``Omega_k^-1`` at the current iterate, i.e. the linearization centre
    """

    def __init__(self, Sigma: List[Arr], Y: Arr, M: Arr) -> None:
        self.Sigma = Sigma
        self.Y = Y
        self.M = M
        self.lmi0 = LMI0Oracle(Sigma)
        self.mk = np.array([float(np.trace(M @ F)) for F in Sigma])

    def assess_optim(self, x: Arr, t: float) -> Tuple[Cut, Optional[float]]:
        """Assess feasibility of ``Omega(x) >= 0`` and optimality of the surrogate.

        :param x: coefficient vector
        :param t: best-so-far objective value
        :return: a ``(cut, value)`` pair
        """
        if cut := self.lmi0.assess_feas(x):
            return cut, None
        R = self.lmi0.ldlt_mgr.sqrt()
        invR = np.linalg.inv(R)
        S = invR @ invR.T
        SY = S @ self.Y
        h = float(np.trace(SY)) + float(x @ self.mk)
        g = np.empty(len(x))
        for i in range(len(x)):
            g[i] = -float(np.trace(S @ self.Sigma[i] @ S @ self.Y)) + self.mk[i]
        if (f := h - t) >= 0:
            return (g, f), None
        return (g, 0.0), h


def omega_of(x: Arr, Sigma: List[Arr]) -> Arr:
    """Return ``Omega(x) = sum_i x_i Sigma_i``.

    :param x: coefficient vector
    :param Sigma: basis matrices
    :return: the assembled matrix
    """
    return sum(c * F for c, F in zip(x, Sigma))


def mle_obj(x: Arr, Sigma: List[Arr], Y: Arr) -> float:
    """Return ``log det Omega(x) + Tr(Omega(x)^-1 Y)``, or ``inf`` if not positive definite.

    :param x: coefficient vector
    :param Sigma: basis matrices
    :param Y: biased sample covariance matrix
    :return: the MLE objective value
    """
    Om = omega_of(x, Sigma)
    sign, logdet = np.linalg.slogdet(Om)
    return np.inf if sign <= 0 else float(logdet + np.trace(np.linalg.solve(Om, Y)))


def cccp_mle(
    Y: Arr,
    Sigma: List[Arr],
    x0: Arr,
    n_outer: int = 40,
    tol: float = 1e-8,
    wrapper: Optional[Callable[[Any], Any]] = None,
) -> Tuple[Arr, int]:
    """Run CCP from ``x0`` until the objective stalls or ``n_outer`` rounds elapse.

    :param Y: biased sample covariance matrix
    :param Sigma: basis matrices
    :param x0: starting coefficient vector
    :param n_outer: maximum number of linearization rounds
    :param tol: objective-change tolerance for early stopping
    :param wrapper: optional ``oracle -> oracle`` hook, e.g. a monotonicity wrapper
    :return: the final coefficient vector and the number of rounds used
    """
    x = np.array(x0, dtype=float)
    f_old = np.inf
    for k in range(n_outer):
        M = np.linalg.inv(omega_of(x, Sigma))
        oracle: Any = cccp_mle_oracle(Sigma, Y, M)
        if wrapper is not None:
            oracle = wrapper(oracle)
        x_new, _, _ = cutting_plane_optim(oracle, Ell(100.0, x), float("inf"))
        if x_new is None:
            return x, k
        f_new = mle_obj(x_new, Sigma, Y)
        if abs(f_old - f_new) < tol:
            return x_new, k + 1
        f_old = f_new
        x = x_new
    return x, n_outer
