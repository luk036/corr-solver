# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple, Union

import numpy as np
from ellalgo.oracles.lmi0_oracle import LMI0Oracle

from .qmi_oracle import QMIOracle

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]

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
    """Oracle for least-squares estimation

    Returns:
        [type]: [description]
    """

    def __init__(self, F: List[Arr], F0: Arr):
        """
        The function initializes the `qmi` and `lmi0` oracles with the given parameters.

        :param F: A list of arrays (F) representing a set of quadratic matrix inequalities (QMIs)
        :type F: List[Arr]
        :param F0: F0 is an array representing the initial feasible solution for the optimization problem
        :type F0: Arr
        """
        self.qmi = QMIOracle(F, F0)
        self.lmi0 = LMI0Oracle(F)

    def assess_optim(self, x: Arr, t: float) -> Tuple[Cut, Optional[float]]:
        """
        The `assess_optim` function assesses the optimality of a given solution `x` and returns a tuple
        containing a cut and an optional float value.

        :param x: The parameter `x` is of type `Arr`, which is likely a numpy array or a list of numbers. It
            represents some input values for the optimization problem
        :type x: Arr
        :param t: The parameter `t` represents the best-so-far optimal value. It is a float value that is
            used in the assessment of the optimization problem
        :type t: float
        :return: The function `assess_optim` returns a tuple containing two elements. The first element is a
            tuple `(g, fj)` which represents a cut and its corresponding objective value. The second element is
            an optional float value `tc` if `fj > 0.0`, otherwise it is `None`.
        """
        n = len(x)
        g = np.zeros(n)

        if cut := self.lmi0.assess_feas(x[:-1]):
            g1, fj = cut
            g[:-1] = g1
            g[-1] = 0.0
            return (g, fj), None

        self.qmi.update(x[-1])
        if cut := self.qmi.assess_feas(x[:-1]):
            g1, fj = cut
            g[:-1] = g1
            self.qmi.ldlt_mgr.witness()
            # n = self.qmi.ldlt_mgr.pos[-1] + 1
            s, n = self.qmi.ldlt_mgr.pos
            wit = self.qmi.ldlt_mgr.wit[s:n]
            g[-1] = -(wit @ wit)
            return (g, fj), None

        g[-1] = 1
        tc = x[-1]
        if (fj := tc - t) > 0.0:
            return (g, fj), None
        return (g, 0.0), tc
