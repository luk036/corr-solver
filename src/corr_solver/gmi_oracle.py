# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import numpy as np
from ellalgo.oracles.ldlt_mgr import LDLTMgr

Cut = Tuple[np.ndarray, float]


# The `GMIOracle` class is an oracle for a General Matrix Inequality constraint, which evaluates the
# function and its negative gradient.
class GMIOracle:
    """Oracle for General Matrix Inequality constraint

            H(x) >= 0

    H.eval(row, col, x): function evalution at (row, col)-element
    H.neggrad[k](rng, x): negative gradient in range rng, the k-term
    """

    def __init__(self, H, m):
        """
        The function initializes an object with attributes H, m, and Q.

        :param H: The parameter `H` is a variable that represents a matrix. It is not clear what the matrix
            represents or how it is used in the code
        :param m: The parameter `m` represents the dimension of the matrix. It is an integer value
        """
        self.H = H
        self.m = m
        self.ldlt_mgr = LDLTMgr(m)

    # def update(self, t):
    #     """
    #     The function "update" updates the value of "self.H" with the value of "t".
    #
    #     :param t: The parameter "t" in the "update" method is a variable that represents the time or the
    #     value that needs to be updated
    #     """
    #     self.H.update(t)

    def assess_feas(self, x: np.ndarray) -> Optional[Cut]:
        """
        The `assess_feas` function assesses the feasibility of a given input `x` and returns a cut if it is
        infeasible, otherwise it returns `None`.

        :param x: An input array of type `np.ndarray`
        :type x: np.ndarray
        :return: The function `assess_feas` returns an optional `Cut` object.
        """

        def get_elem(row, col):
            """
            The function `get_elem` returns the evaluation of the function `H` at the given indices `i` and `j`,
            with the input `x`.

            :param i: The parameter "i" represents the row index of the element in the matrix
            :param j: The parameter "j" represents the column index of the element in the matrix
            :return: The function `get_elem` is returning the result of calling the `eval` method on the `H`
                object with the arguments `row`, `col`, and `x`.
            """
            return self.H.eval(row, col, x)

        if self.ldlt_mgr.factor(get_elem):
            return None
        ep = self.ldlt_mgr.witness()
        g = self.H.neg_grad_sym_quad(self.ldlt_mgr, x)
        return g, ep
