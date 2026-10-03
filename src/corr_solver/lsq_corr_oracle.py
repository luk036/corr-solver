# -*- coding: utf-8 -*-
"""
Least-squares correlation oracle.

Solves ``min ||F0 - F(x)|| s.t. F(x) >= 0`` for a matrix ``F(x)`` built from
basis matrices ``F[k]``. Feasibility is checked through the ``lmi0`` oracle,
then the ``qmi`` oracle for the quadratic-matrix-inequality reformulation;
optimality is assessed against the best-so-far value ``t``.
"""

from typing import List, Optional

import numpy as np

from .backends import default_backend
from .protocols import OracleBackend
from .types import Arr, Assessment


#    min   ‖ F0 − F(x) ‖
#    s.t.  F(x) ⪰ 0
#
#    Transform the problem into:
#
#    min   t
#    s.t.  x[n+1] ≤ t
#          x[n+1]*I − F(x)^T F(x) ⪰ 0
#
#    where:
#    1. F(x) = F[1] x[1] + ··· + F[n] x[n]
#    2. {Fk}i,j = Ψk(‖sj − si‖)
class lsq_oracle:
    """Oracle for least-squares estimation.

    Solves the problem: min ||F0 - F(x)|| s.t. F(x) >= 0

    The oracle transforms the problem into::

        min t
        s.t. x[n+1] <= t
             x[n+1]*I - F(x)^T F(x) >= 0

    where ``F(x) = F[1] x[1] + ... + F[n] x[n]`` and ``{Fk}i,j = Ψk(||sj - si||)``
    """

    def __init__(self, F: List[Arr], F0: Arr, backend: Optional[OracleBackend] = None):
        if backend is None:
            backend = default_backend()
        self.qmi = backend.qmi(F, F0)
        self.lmi0 = backend.lmi0(F)

    def assess_optim(self, x: Arr, t: float) -> Assessment:
        """Assess optimality of ``x`` against the best-so-far value ``t``.

        :param x: augmented variable vector ``(coeffs..., t)``
        :param t: best-so-far optimal value
        :return: an :class:`~corr_solver.types.Assessment`
        """
        n = len(x)
        g = np.zeros(n)

        if cut := self.lmi0.assess_feas(x[:-1]):
            g1, fj = cut
            g[:-1] = g1
            g[-1] = 0.0
            return Assessment((g, fj), None)

        self.qmi.update(x[-1])
        if cut := self.qmi.assess_feas(x[:-1]):
            g1, fj = cut
            g[:-1] = g1
            g[-1] = -self.qmi.witness_sq()
            return Assessment((g, fj), None)
        g[-1] = 1
        tc = x[-1]
        if (fj := tc - t) > 0.0:
            return Assessment((g, fj), None)
        return Assessment((g, 0.0), tc)


LSQOracle = lsq_oracle
