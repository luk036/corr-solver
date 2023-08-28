# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple, Union

import numpy as np

from .gmi_oracle import GMIOracle

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]

# import cholutil


class QMIOracle:
    class QMI:
        """Oracle for Quadratic Matrix Inequality

        The QMI class represents an oracle for solving a quadratic matrix inequality problem.

          find  x
          s.t.  t*I - F(x)' F(x) ⪰ 0

        where

          F(x) = F0 - (F1 * x1 + F2 * x2 + ...)
        """

        t = None
        count = 0

        def __init__(self, F: List[Arr], F0: Arr):
            """
            The function initializes the variables F, F0, and Fx with the given arguments.

            :param F: F is a list of arrays. Each array in the list represents a feature vector. The
            feature vectors can have different lengths, but they all have the same number of columns
            :type F: List[Arr]
            :param F0: F0 is a 2-dimensional array (matrix) representing the initial state of the
            system. It has n rows and m columns
            :type F0: Arr
            """
            self.F = F
            self.F0 = F0
            n, m = F0.shape
            self.Fx = np.zeros([m, n])

        def update(self, t: float):
            """
            The `update` function updates the value of `self.t` with the input `t`.

            :param t: The parameter `t` represents the best-so-far optimal value
            :type t: float
            """
            self.t = t

        def eval(self, i, j, x: Arr) -> float:
            """
            The `eval` function calculates a value based on the given parameters and returns it.

            :param i: The parameter `i` represents the index of the first dimension of the arrays `Fx` and `F0`.
            It is used to access specific elements of these arrays
            :param j: The parameter `j` represents the index of the column in the matrix `self.Fx` that is being
            evaluated
            :param x: The parameter `x` is an array of values
            :type x: Arr
            :return: a float value. If `i` is equal to `j`, it returns `self.t + a`, otherwise it returns `a`.
            """
            if i < j:
                raise AssertionError()
            if self.count < i + 1:
                nx = len(x)
                self.count = i + 1
                self.Fx[i] = self.F0[:, i]
                self.Fx[i] -= sum(self.F[k][:, i] * x[k] for k in range(nx))
            a = -(self.Fx[i] @ self.Fx[j])
            if i == j:
                return self.t + a
            return a

        def neg_grad_sym_quad(self, Q, x: Arr):
            """
            The function `neg_grad_sym_quad` calculates the negative gradient of a symmetric quadratic function.

            :param Q: Q is a quadratic matrix represented as a sparse matrix. It has two attributes: p and v. p
            is a tuple representing the starting and ending indices of the non-zero elements in the matrix, and
            v is a numpy array representing the values of the non-zero elements
            :param x: The parameter `x` is an array
            :type x: Arr
            :return: the gradient vector `g`.
            """
            s, n = Q.pos
            v = Q.v[s:n]
            Av = v @ self.Fx[s:n]
            g = np.array([-2 * ((v @ Fk[s:n]) @ Av) for Fk in self.F])
            return g

    def __init__(self, F, F0):
        """
        The function initializes an object with attributes qmi, gmi, and Q based on the input arguments F
        and F0.

        :param F: A list of arrays. Each array represents a feature matrix for a different class. The
        feature matrix has shape (n, m), where n is the number of samples and m is the number of features
        :param F0: F0 is a 2-dimensional array representing the reference distribution. It has n rows and m
        columns
        """
        n, m = F0.shape
        self.qmi = self.QMI(F, F0)
        self.gmi = GMIOracle(self.qmi, m)
        self.ldlt_mgr = self.gmi.ldlt_mgr

    def update(self, t: float):
        """
        The function updates the best-so-far optimal value.

        :param t: The parameter `t` represents the best-so-far optimal value
        :type t: float
        """
        self.qmi.update(t)

    def assess_feas(self, x: Arr) -> Optional[Cut]:
        """
        The `assess_feas` function assesses the feasibility of a given input and returns a cut if it is
        feasible.

        :param x: An array of values
        :type x: Arr
        :return: an Optional[Cut] object.
        """
        self.qmi.count = 0
        return self.gmi.assess_feas(x)
