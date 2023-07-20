# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Union

import numpy as np
from scipy.interpolate import BSpline

from .corr_oracle import construct_distance_matrix

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


def mono_oracle(x):
    """
    The `mono_oracle` function is an oracle that checks if a given array `x` satisfies the monotonic
    decreasing constraint and returns the gradient and the first violation if it exists.
    
    :param x: The parameter `x` is a list or array of numbers. It represents a sequence of values that
    we want to check for the monotonic decreasing constraint
    :return: The function `mono_oracle` returns two values: `g` and `fj`. `g` is a numpy array of zeros
    with the same length as `x`, where elements are set to -1.0 and 1.0 to enforce the monotonic
    decreasing constraint. `fj` is the difference between the next element and the current element in
    `x`.
    """
    # monotonic decreasing constraint
    n = len(x)
    g = np.zeros(n)
    for i in range(n - 1):
        if (fj := x[i + 1] - x[i]) > 0.0:
            g[i] = -1.0
            g[i + 1] = 1.0
            return g, fj


# The `mono_decreasing_oracle2` class is an oracle that checks if a given sequence is monotonically
# decreasing.
class mono_decreasing_oracle2:
    """oracle for monotonic decreasing constraint

    Returns:
        [type]: [description]
    """

    def __init__(self, basis):
        """
        The function initializes an object with a given basis.
        
        :param basis: The `basis` parameter is a variable that is passed to the `__init__` method of a
        class. It is used to initialize the `basis` attribute of the class instance. The `basis` attribute
        can then be accessed and used throughout the class methods
        """
        self.basis = basis

    def assess_optim(self, x: Arr, t: float) -> Tuple[Cut, Optional[float]]:
        """
        The function assess_optim assesses the optimality of a given solution by checking if it satisfies a
        monotonic decreasing constraint, and if not, it calls another function to assess optimality.
        
        :param x: An array of values
        :type x: Arr
        :param t: The parameter `t` represents the best-so-far optimal value. It is a float value that is
        used in the function to assess the optimality of a solution
        :type t: float
        :return: The function `assess_optim` returns a tuple containing a `Cut` object and an optional float
        value.
        """
        # monotonic decreasing constraint
        n = len(x)
        g = np.zeros(n)
        if cut := mono_oracle(x[:-1]):
            g1, fj = cut
            g[:-1] = g1
            g[-1] = 0.0
            return (g, fj), None
        return self.basis.assess_optim(x, t)


def corr_bspline(Y, s, m, oracle, corr_core):
    """
    The `corr_bspline` function takes in input parameters `Y`, `s`, `m`, `oracle`, and `corr_core`, and
    returns a BSpline object, the number of iterations, and a feasibility indicator.
    
    :param Y: The input data Y for the B-spline algorithm
    :param s: The parameter `s` represents the number of control points in the B-spline curve. It
    determines the flexibility and smoothness of the curve
    :param m: The parameter `m` represents the number of control points in the B-spline curve. It
    determines the flexibility and smoothness of the curve. A higher value of `m` will result in a more
    flexible curve that can better fit the data, but it may also lead to overfitting
    :param oracle: The `oracle` parameter is a function that takes in the signal `Sig` and the observed
    data `Y` and returns the predicted values `Pb`. It is used to generate the predicted values for the
    given signal and observed data
    :param corr_core: The `corr_core` parameter is a function that takes in the following arguments:
    :return: The function `corr_bspline` returns three values:
    """
    Sig, t, k = generate_bspline_info(s, m)
    Pb = oracle(Sig, Y)
    omega = mono_decreasing_oracle2(Pb)
    c, num_iters, feasible = corr_core(Y, m, omega)
    return BSpline(t, c, k), num_iters, feasible


def generate_bspline_info(s, m):
    """
    The function `generate_bspline_info` generates B-spline information given a set of points and a
    desired number of B-splines.
    
    :param s: The parameter `s` is a list or array of data points that define the shape or curve that
    you want to approximate using B-splines
    :param m: The parameter `m` represents the number of B-spline basis functions to generate. It
    determines the number of basis functions that will be used to approximate the input data
    :return: The function `generate_bspline_info` returns three values: `Sig`, `t`, and `k`.
    """
    k = 2  # quadratic bspline
    h = s[-1] - s[0]
    d = np.sqrt(h @ h)
    t = np.linspace(0, d * 1.2, m + k + 1)
    spls = []
    for i in range(m):
        coeff = np.zeros(m)
        coeff[i] = 1
        spls += [BSpline(t, coeff, k)]
    D = construct_distance_matrix(s)
    Sig = []
    for i in range(m):
        Sig += [spls[i](D)]
    return Sig, t, k
