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

from .math_utils import inverse_sqrt_gram, mle_obj, omega_of
from .mle_corr_oracle import MleOptimOracle
from .solvers import SolverConfig
from .types import Arr, Cut


class cccp_mle_oracle(MleOptimOracle):
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

    def _first_cut(self, x: Arr) -> Optional[Cut]:
        return self.lmi0.assess_feas(x)

    def _sqrt(self) -> Arr:
        return self.lmi0.ldlt_mgr.sqrt()

    def _value_and_grad(self, x: Arr, R: Arr) -> Tuple[float, Arr]:
        S, SY = inverse_sqrt_gram(R, self.Y)
        h = float(np.trace(SY)) + float(x @ self.mk)
        g = np.empty(len(x))
        for i in range(len(x)):
            g[i] = -float(np.trace(S @ self.Sigma[i] @ S @ self.Y)) + self.mk[i]
        return h, g


def cccp_mle(
    Y: Arr,
    Sigma: List[Arr],
    x0: Arr,
    n_outer: int = 40,
    tol: float = 1e-8,
    wrapper: Optional[Callable[[Any], Any]] = None,
    config: Optional[SolverConfig] = None,
) -> Tuple[Arr, int]:
    """Run CCP from ``x0`` until the objective stalls or ``n_outer`` rounds elapse.

    :param Y: biased sample covariance matrix
    :param Sigma: basis matrices
    :param x0: starting coefficient vector
    :param n_outer: maximum number of linearization rounds
    :param tol: objective-change tolerance for early stopping
    :param wrapper: optional ``oracle -> oracle`` hook, e.g. a monotonicity wrapper
    :param config: solver constants; defaults to :class:`SolverConfig`
    :return: the final coefficient vector and the number of rounds used
    """
    config = SolverConfig() if config is None else config
    x = np.array(x0, dtype=float)
    f_old = np.inf
    for k in range(n_outer):
        M = np.linalg.inv(omega_of(x, Sigma))
        oracle: Any = cccp_mle_oracle(Sigma, Y, M)
        if wrapper is not None:
            oracle = wrapper(oracle)
        x_new, _, _ = cutting_plane_optim(oracle, Ell(config.cccp_r0, x), float("inf"))
        if x_new is None:
            return x, k
        f_new = mle_obj(x_new, Sigma, Y)
        if abs(f_old - f_new) < tol:
            return x_new, k + 1
        f_old = f_new
        x = x_new
    return x, n_outer


CCCPMLEOracle = cccp_mle_oracle
