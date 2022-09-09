# -*- coding: utf-8 -*-
# import cvxpy as cvx
from typing import Optional, Tuple, Union

import numpy as np
from lmi_solver.lmi0_oracle import LMI0Oracle
from lmi_solver.lmi_oracle import LMIOracle

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


class mle_oracle:
    def __init__(self, Sig: Arr, Y: Arr):
        """Maximum likelyhood estimation:

            min  log det Ω(p) + Tr( Ω(p)^{-1} Y )
            s.t. 2Y ⪰ Ω(p) ⪰ 0,

        Arguments:
            Sig (Arr): Covariance matrix
            Y (Arr): Biased covariance matrix
        """
        self.Y = Y
        self.Sig = Sig
        self.lmi0 = LMI0Oracle(Sig)
        self.lmi = LMIOracle(Sig, 2 * Y)
        # self.lmi2 = LMI2Oracle(Sig, 2*Y)

    def assess_optim(self, x: Arr, t: float) -> Tuple[Cut, Optional[float]]:
        """[summary]

        Arguments:
            x (Arr): coefficients of basis functions
            t (float): the best-so-far optimal value

        Returns:
            Tuple[Cut, float]: [description]
        """
        if cut := self.lmi.assess_feas(x):
            return cut, None

        if cut := self.lmi0.assess_feas(x):
            return cut, None

        R = self.lmi0.Q.sqrt()
        invR = np.linalg.inv(R)
        S = invR @ invR.T
        SY = S @ self.Y
        diag = np.diag(R)
        f1 = 2 * np.sum(np.log(diag)) + np.trace(SY)

        n = len(x)
        m = len(self.Y)
        g = np.zeros(n)
        for i in range(n):
            SFsi = S @ self.Sig[i]
            # g[i] = sum(S[k] @ self.Sig[k] for k in range(m))
            g[i] = np.trace(SFsi)
            g[i] -= sum(SFsi[k, :] @ SY[:, k] for k in range(m))

        f = f1 - t
        if (f := f1 - t) >= 0:
            return (g, f), None
        return (g, 0.0), f1
