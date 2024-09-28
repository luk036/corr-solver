# -*- coding: utf-8 -*-
"""
Least Squares Correlation Oracle

This code defines a class called lsq_oracle (least squares oracle) that is designed to solve a specific type of optimization problem. The purpose of this code is to find the best solution to a mathematical problem involving matrices and inequalities.

The lsq_oracle class takes two inputs when it's created: a list of arrays called F and another array called F0. These inputs represent mathematical objects that define the problem the oracle is trying to solve.

The main output of this code is produced by the assess_optim method. This method takes two inputs: an array x (which represents a potential solution) and a float t (which represents the best solution found so far). It returns information about whether the given solution is feasible and optimal, and if not, it provides guidance on how to improve the solution.

The code achieves its purpose through a two-step process:

1. First, it checks if the solution satisfies certain constraints using the lmi0 oracle.
2. If that passes, it then checks another set of constraints using the qmi oracle.

If either of these checks fail, the method returns information about how the solution violates the constraints. If both checks pass, it compares the current solution to the best solution found so far.

An important transformation happening in this code is the conversion of the original problem into a slightly different form. Instead of directly minimizing the difference between F0 and F(x), it introduces a new variable t and tries to minimize t while ensuring that certain conditions involving t are met. This transformation allows the problem to be solved using standard optimization techniques.

The code uses some advanced mathematical concepts (like matrix inequalities) that might be challenging for a beginner to fully understand. However, the overall structure of the code follows a logical flow: initialize the problem, check constraints, and either return information about constraint violations or assess the optimality of the solution.

In simple terms, you can think of this code as a smart calculator that's trying to solve a complex math problem. It takes in the problem description (F and F0), tries different solutions (x), and tells you whether each solution is good or not, and how to make it better if it's not good enough.
"""

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
