# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import numpy as np
from lmi_solver.chol_ext import LDLTMgr

Cut = Tuple[np.ndarray, float]


class GMIOracle:
    """Oracle for General Matrix Inequality constraint

            H(x) >= 0

    H.eval(i, j, x): function evalution at (i,j)-element
    H.neggrad[k](p, x): negative gradient in range p, the k-term

    """

    def __init__(self, H, m):
        """[summary]

        Arguments:
            H ([type]): [description]
            n (int): dimension
        """
        self.H = H
        self.m = m
        self.Q = LDLTMgr(m)

    def update(self, t):
        self.H.update(t)

    def assess_feas(self, x: np.ndarray) -> Optional[Cut]:
        """[summary]

        Arguments:
            x (np.ndarray): [description]

        Returns:
            Optional[Cut]: [description]
        """

        def get_elem(i, j):
            return self.H.eval(i, j, x)

        if self.Q.factor(get_elem):
            return None
        ep = self.Q.witness()
        g = self.H.neg_grad_sym_quad(self.Q, x)
        return g, ep
