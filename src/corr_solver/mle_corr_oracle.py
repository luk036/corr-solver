"""
Maximum-likelihood estimation oracle.

Fits coefficients ``p`` of ``Omega(p)`` to a biased sample covariance matrix
``Y`` by minimizing ``log det Omega(p) + Tr(Omega(p)^{-1} Y)`` subject to
``2Y >= Omega(p) >= 0``. Feasibility is checked with the ``lmi`` and ``lmi0``
oracles; the objective and its gradient are then compared against the

best-so-far value ``t``.






"""

from typing import List, Optional, Tuple

import numpy as np
from ellalgo.oracles.lmi0_oracle import LMI0Oracle
from ellalgo.oracles.lmi_oracle import LMIOracle

Cut = Tuple[np.ndarray, float]


# The `mle_oracle` class represents an oracle for maximum likelihood estimation, which minimizes a
# certain objective function subject to linear matrix inequality constraints.
class mle_oracle:
    def __init__(self, Sigma: List[np.ndarray], Y: np.ndarray):
        """Maximum likelyhood estimation:

            min  log det Ω(p) + Tr( Ω(p)^{-1} Y )
            s.t. 2Y ⪰ Ω(p) ⪰ 0,


        """
        self.Y = Y
        self.Sigma = Sigma
        self.lmi0 = LMI0Oracle(Sigma)
        self.lmi = LMIOracle(Sigma, 2 * Y)

    def assess_optim(self, x: np.ndarray, t: float) -> Tuple[Cut, Optional[float]]:
        """
        The `assess_optim` function assesses the feasibility and optimality of a given solution by
        calculating various values and returning a tuple of cuts and a float value.

        :param x: The parameter `x` is a numpy array representing the coefficients of basis functions. It is
            used as input to assess the feasibility of a solution
        :type x: Arr
        :param t: The parameter `t` represents the best-so-far optimal value. It is a float value that is
            used in the calculation of the objective function `f`
        :type t: float
        :return: The function `assess_optim` returns a tuple containing two elements. The first element is a
            `Cut` object or a tuple `(g, f)` depending on the condition. The second element is either `None` or
            a float value.
        """
        if cut := self.lmi.assess_feas(x):
            return cut, None

        if cut := self.lmi0.assess_feas(x):
            return cut, None

        R = self.lmi0.ldlt_mgr.sqrt()
        invR = np.linalg.inv(R)
        S = invR @ invR.T
        SY = S @ self.Y
        diag = np.diag(R)
        f1 = 2 * np.sum(np.log(diag)) + np.trace(SY)

        n = len(x)
        g = np.zeros(n)
        # g[i] = tr(S @ Sigma[i]) - tr(S @ Sigma[i] @ SY)
        #      = tr((S - SY @ S) @ Sigma[i])      [cyclic perm of trace]
        #      = sum((S - SY @ S).T * Sigma[i])    [Frobenius inner prod]
        V = S - SY @ S  # pre-compute once, n×n
        for i in range(n):
            g[i] = np.sum(V.T * self.Sigma[i])

        if (f := f1 - t) >= 0:
            return (g, f), None
        return (g, 0.0), f1
