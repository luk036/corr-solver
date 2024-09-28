"""
mle_oracle

This code defines a class called mle_oracle which is designed to solve a maximum likelihood estimation problem. The purpose of this code is to find the best parameters for a statistical model based on observed data, while satisfying certain constraints.

The mle_oracle class takes two inputs when initialized: Sigma and Y. Sigma represents a covariance matrix, which describes how different variables in a dataset are related to each other. Y is a biased sample covariance matrix, which is an estimate of the true covariance based on observed data.

The main output of this class is produced by the assess_optim method. This method takes two inputs: x (a set of coefficients) and t (the best optimal value found so far). It returns a tuple containing information about whether the current solution is feasible and optimal, along with some additional values used in the optimization process.

To achieve its purpose, the code uses a technique called linear matrix inequality (LMI) optimization. It creates two LMI oracles (lmi0 and lmi) which are used to check if the current solution satisfies certain constraints. The assess_optim method first checks if the solution is feasible using these oracles. If it's not feasible, it returns information about why it's not feasible.

If the solution is feasible, the method then calculates a value f1, which represents the objective function of the maximum likelihood estimation problem. This calculation involves matrix operations like inversion, multiplication, and calculating traces and determinants. The method also computes a gradient g, which indicates how the objective function changes with respect to small changes in the input x.

Finally, the method compares the calculated f1 with the input t to determine if a better solution has been found. If f1 is better than t, it returns this new value along with the gradient. Otherwise, it returns information that can be used to continue the optimization process.

The important logic flows in this code include the feasibility checks, the calculation of the objective function and its gradient, and the comparison of the current solution with the best known solution. The data transformations mainly involve matrix operations on the input covariance matrices.

Overall, this code provides a way to solve a complex statistical optimization problem by iteratively improving a solution while ensuring it satisfies certain constraints. It's a building block that would typically be used as part of a larger optimization algorithm.
"""

from typing import Optional, Tuple, Union

import numpy as np
from ellalgo.oracles.lmi0_oracle import LMI0Oracle
from ellalgo.oracles.lmi_oracle import LMIOracle

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


# The `mle_oracle` class represents an oracle for maximum likelihood estimation, which minimizes a
# certain objective function subject to linear matrix inequality constraints.
class mle_oracle:
    def __init__(self, Sigma: Arr, Y: Arr):
        """Maximum likelyhood estimation:

            min  log det Ω(p) + Tr( Ω(p)^{-1} Y )
            s.t. 2Y ⪰ Ω(p) ⪰ 0,

        The function initializes an object with given covariance matrix and biased covariance matrix,
        and creates LMI oracles for optimization.

        :param Sigma: The parameter "Sigma" represents the covariance matrix, which is a square matrix that
            describes the variances and covariances of a set of random variables. It is used in the maximum
            likelihood estimation algorithm to estimate the parameters of a statistical model
        :type Sigma: Arr
        :param Y: The parameter Y represents a biased sample covariance matrix. It is used in the maximum
            likelihood estimation problem to constrain the covariance matrix Ω(p) such that 2Y is greater
            than or equal to Ω(p)
        :type Y: Arr
        """
        self.Y = Y
        self.Sigma = Sigma
        self.lmi0 = LMI0Oracle(Sigma)
        self.lmi = LMIOracle(Sigma, 2 * Y)

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

        R = self.lmi0.ldlt_mgr.sqrt()
        invR = np.linalg.inv(R)
        S = invR @ invR.T
        SY = S @ self.Y
        diag = np.diag(R)
        f1 = 2 * np.sum(np.log(diag)) + np.trace(SY)

        n = len(x)
        m = len(self.Y)
        g = np.zeros(n)
        for i in range(n):
            SFsi = S @ self.Sigma[i]
            g[i] = np.trace(SFsi)
            g[i] -= sum(SFsi[k, :] @ SY[:, k] for k in range(m))

        if (f := f1 - t) >= 0:
            return (g, f), None
        return (g, 0.0), f1
