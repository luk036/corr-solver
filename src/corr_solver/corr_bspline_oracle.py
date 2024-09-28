"""
Corr_bspline_oracle.py

This code is designed to work with B-splines, which are mathematical functions used to create smooth curves. The main purpose of this code is to generate a B-spline curve that fits a given set of data points while satisfying certain constraints, particularly a monotonic decreasing constraint.

The code takes several inputs, including data points (Y), the number of control points for the B-spline (site and m), an oracle function for optimization, and a core correlation function (corr_core). These inputs are used to create and optimize the B-spline curve.

The main output of this code is a B-spline object, which represents the fitted curve. It also returns the number of iterations taken to optimize the curve and a boolean indicating whether the optimization was successful (feasible).

To achieve its purpose, the code uses several functions and a class:

1. The mono_oracle function checks if a given sequence of numbers is monotonically decreasing (each number is less than or equal to the previous one). If it finds a violation, it returns information about where the violation occurred.

2. The mono_decreasing_oracle2 class wraps the mono_oracle function and provides an interface for the optimization process.

3. The corr_bspline function is the main function that ties everything together. It generates the B-spline information, sets up the optimization problem, and calls the core correlation function to optimize the B-spline coefficients.

4. The generate_bspline_info function creates the necessary information for constructing a B-spline, including the knot vector (t) and the basis functions (Sigma).

The code follows this general flow:

1. Generate the B-spline information (knots, basis functions)
2. Set up the optimization problem with constraints
3. Optimize the B-spline coefficients
4. Create and return the final B-spline object

An important part of the logic is the monotonic decreasing constraint, which ensures that the resulting B-spline curve always decreases (or stays flat) as it moves from left to right. This is enforced through the mono_oracle and mono_decreasing_oracle2 functions.

The code also performs some data transformations, such as converting the input site points into a distance matrix, and creating B-spline basis functions. These transformations help in setting up the optimization problem and constructing the final B-spline curve.

Overall, this code provides a way to fit a smooth, monotonically decreasing curve to a set of data points, which can be useful in various applications such as data analysis, signal processing, or computer graphics.
"""

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


def corr_bspline(Y, site, m, oracle, corr_core):
    """
    The `corr_bspline` function takes in input parameters `Y`, `site`, `m`, `oracle`, and `corr_core`, and
    returns a BSpline object, the number of iterations, and a feasibility indicator.

    :param Y: The input data Y for the B-spline algorithm
    :param site: The parameter `site` represents the number of control points in the B-spline curve. It
        determines the flexibility and smoothness of the curve
    :param m: The parameter `m` represents the number of control points in the B-spline curve. It
        determines the flexibility and smoothness of the curve. A higher value of `m` will result in a more
        flexible curve that can better fit the data, but it may also lead to overfitting
    :param oracle: The `oracle` parameter is a separation oracle
    :param corr_core: The `corr_core` parameter is a function that takes in the following arguments:
    :return: The function `corr_bspline` returns three values:
    """
    Sigma, t, k = generate_bspline_info(site, m)
    Pb = oracle(Sigma, Y)
    omega = mono_decreasing_oracle2(Pb)
    c, num_iters, feasible = corr_core(Y, m, omega)
    return BSpline(t, c, k), num_iters, feasible


def generate_bspline_info(site, m):
    """
    The function `generate_bspline_info` generates B-spline information given a set of points and a
    desired number of B-splines.

    :param site: The parameter `site` is a list or array of data points that define the shape or curve that
        you want to approximate using B-splines
    :param m: The parameter `m` represents the number of B-spline basis functions to generate. It
        determines the number of basis functions that will be used to approximate the input data
    :return: The function `generate_bspline_info` returns three values: `Sigma`, `t`, and `k`.
    """
    k = 2  # quadratic bspline
    h = site[-1] - site[0]
    d = np.sqrt(h @ h)
    t = np.linspace(0, d * 1.2, m + k + 1)
    spls = []
    for i in range(m):
        coeff = np.zeros(m)
        coeff[i] = 1
        spls += [BSpline(t, coeff, k)]
    D = construct_distance_matrix(site)
    Sigma = []
    for i in range(m):
        Sigma += [spls[i](D)]
    return Sigma, t, k
