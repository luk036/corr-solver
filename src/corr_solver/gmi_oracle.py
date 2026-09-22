# -*- coding: utf-8 -*-
from typing import Any, Optional

import numpy as np
from ellalgo.oracles.ldlt_mgr import LDLTMgr

from .types import Cut


# The `GMIOracle` class is an oracle for a General Matrix Inequality constraint, which evaluates the
# function and its negative gradient.
class GMIOracle:
    """Oracle for General Matrix Inequality constraint

            H(x) >= 0

    H.eval(row, col, x): function evalution at (row, col)-element
    H.neggrad[k](rng, x): negative gradient in range rng, the k-term
    """

    def __init__(self, H: Any, m: int) -> None:
        self.H = H
        self.m = m
        self.ldlt_mgr = LDLTMgr(m)

    def assess_feas(self, x: np.ndarray) -> Optional[Cut]:
        """
        The `assess_feas` function assesses the feasibility of a given input `x` and returns a cut if it is
        infeasible, otherwise it returns `None`.

        :param x: An input array of type `np.ndarray`
        :type x: np.ndarray
        :return: The function `assess_feas` returns an optional `Cut` object.
        """

        def get_elem(row: int, col: int) -> float:
            return self.H.eval(row, col, x)

        if self.ldlt_mgr.factor(get_elem):
            return None
        ep = self.ldlt_mgr.witness()
        g = self.H.neg_grad_sym_quad(self.ldlt_mgr, x)
        return g, ep
