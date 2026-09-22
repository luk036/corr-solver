# -*- coding: utf-8 -*-
from typing import Any, List, Optional

import numpy as np

from .gmi_oracle import GMIOracle
from .types import Arr, Cut


class QMIOracle:
    class QMI:
        """Oracle for Quadratic Matrix Inequality

        The QMI class represents an oracle for solving a quadratic matrix inequality problem.

          find  x
          s.t.  t*I - F(x)' F(x) ⪰ 0

        where

          F(x) = F0 - (F1 * x1 + F2 * x2 + ...)
        """

        def __init__(self, F: List[Arr], F0: Arr):
            """
            :param F: feature vectors, each with the same number of columns
            :type F: List[Arr]
            :param F0: initial state matrix with n rows and m columns
            :type F0: Arr
            """
            self.t = 0.0
            self.count = 0
            self.F = F
            self.F0 = F0
            n, m = F0.shape
            self.Fx = np.zeros([m, n])

        def update(self, t: float) -> None:
            self.t = t

        def eval(self, row: int, col: int, x: Arr) -> float:
            """
            :param row: row index in the matrix
            :param col: column index in the matrix
            :param x: variable vector
            :type x: Arr
            """
            if row < col:
                raise AssertionError()
            if self.count < row + 1:
                nx = len(x)
                self.count = row + 1
                self.Fx[row] = self.F0[:, row]
                self.Fx[row] -= sum(self.F[k][:, row] * x[k] for k in range(nx))
            a = float(-(self.Fx[row] @ self.Fx[col]))
            if row == col:
                return self.t + a
            return a

        def neg_grad_sym_quad(self, Q: Any, _: Arr) -> np.ndarray:
            """
            :param Q: sparse quadratic matrix with attributes ``p`` (index range) and ``v`` (nonzero values)
            :param _: unused placeholder
            :type _: Arr
            """
            s, n = Q.pos
            v = Q.wit[s:n]
            Av = v @ self.Fx[s:n]
            g = np.array([-2 * ((v @ Fk[s:n]) @ Av) for Fk in self.F])
            return g

    def __init__(self, F: List[Arr], F0: Arr) -> None:
        """
        :param F: feature matrices, each of shape (n, m)
        :param F0: reference distribution matrix (n rows, m columns)
        """
        _, m = F0.shape
        self.qmi = self.QMI(F, F0)
        self.gmi = GMIOracle(self.qmi, m)
        self.ldlt_mgr = self.gmi.ldlt_mgr

    def update(self, t: float) -> None:
        self.qmi.update(t)

    def assess_feas(self, x: Arr) -> Optional[Cut]:
        """
        :param x: candidate point
        :type x: Arr
        :return: feasibility cut if x is infeasible, None otherwise
        """
        self.qmi.count = 0
        return self.gmi.assess_feas(x)
