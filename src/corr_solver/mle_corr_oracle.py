# -*- coding: utf-8 -*-
# import cvxpy as cvx
from typing import Optional, Tuple, Union

import numpy as np
from ellalgo.oracles.lmi0_oracle import LMI0Oracle
from ellalgo.oracles.lmi_oracle import LMIOracle

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


# The `mle_oracle` class represents an oracle for maximum likelihood estimation, which minimizes a
# certain objective function subject to linear matrix inequality constraints.
class mle_oracle:
    def __init__(self, Sig: Arr, Y: Arr):
        """Maximum likelyhood estimation:

            min  log det Ω(p) + Tr( Ω(p)^{-1} Y )
            s.t. 2Y ⪰ Ω(p) ⪰ 0,

        The function initializes an object with given covariance matrix and biased covariance matrix,
        and creates LMI oracles for optimization.

        :param Sig: The parameter "Sig" represents the covariance matrix, which is a square matrix that
        describes the variances and covariances of a set of random variables. It is used in the maximum
        likelihood estimation algorithm to estimate the parameters of a statistical model
        :type Sig: Arr
        :param Y: The parameter Y represents a biased covariance matrix. It is used in the maximum
        likelihood estimation problem to constrain the covariance matrix Ω(p) such that 2Y is greater
        than or equal to Ω(p)
        :type Y: Arr
        """
        self.Y = Y
        self.Sig = Sig
        self.lmi0 = LMI0Oracle(Sig)
        self.lmi = LMIOracle(Sig, 2 * Y)
        # self.lmi2 = LMI2Oracle(Sig, 2*Y)

    def assess_optim(self, x: Arr, t: float) -> Tuple[Cut, Optional[float]]:
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
