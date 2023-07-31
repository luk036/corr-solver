from typing import List, Tuple

import numpy as np
from lds_py.lds import Halton

Arr = np.ndarray
Cut = Tuple[Arr, float]


def create_2d_sites(nx=10, ny=8) -> Arr:
    """
    The function `create_2d_sites` generates a 2D array of site locations using the Halton sequence.

    :param nx: The parameter `nx` represents the number of sites in the x-direction, while `ny`
    represents the number of sites in the y-direction, defaults to 10 (optional)
    :param ny: The parameter `ny` represents the number of rows in the 2D sites object, defaults to 8
    (optional)
    :return: The function `create_2d_sites` returns a 2D array representing the location of sites.
    """
    n = nx * ny
    s_end = np.array([10.0, 8.0])
    hgen = Halton([2, 3])
    s = s_end * np.array([hgen.pop() for _ in range(n)])
    return s


def create_2d_isotropic(s: Arr, N=3000) -> Arr:
    """
    The function `create_2d_isotropic` generates a biased covariance matrix for a 2D isotropic object
    based on the location of sites.

    :param s: The parameter `s` is the location of sites. It is expected to be a 2D array where each row
    represents the coordinates of a site
    :type s: Arr
    :param N: The parameter N represents the number of iterations or samples used to create the 2D
    isotropic object. It determines the number of times the loop runs to generate random values and
    calculate the outer product. The larger the value of N, the more accurate the estimation of the
    biased covariance matrix will be,, defaults to 3000 (optional)
    :return: The function `create_2d_isotropic` returns a biased covariance matrix `Y`.
    """
    n = s.shape[0]
    sdkern = 0.12  # width of kernel
    var = 2.0  # standard derivation
    tau = 0.00001  # standard derivation of white noise
    np.random.seed(5)

    Sig = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            d = np.array(s[j]) - np.array(s[i])
            Sig[i, j] = np.exp(-sdkern * (d.dot(d)))
            Sig[j, i] = Sig[i, j]

    A = np.linalg.cholesky(Sig)
    Y = np.zeros((n, n))

    for _ in range(N):
        x = var * np.random.randn(n)
        y = A @ x + tau * np.random.randn(n)
        Y += np.outer(y, y)

    Y /= N
    return Y


def construct_distance_matrix(s: Arr) -> Arr:
    """
    The function `construct_distance_matrix` takes in a list of site locations and returns a distance
    matrix object where each element represents the distance between two sites.

    :param s: The parameter `s` is the location of sites. It is an array that contains the coordinates
    of each site
    :type s: Arr
    :return: a distance matrix object.
    """
    n = len(s)
    D1 = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            h = s[j] - s[i]
            d = np.sqrt(h @ h)
            D1[i, j] = d
            D1[j, i] = d
    return D1


def construct_poly_matrix(s: Arr, m) -> List[Arr]:
    """
    The function `construct_poly_matrix` takes in a list of site locations `s` and a degree `m`, and
    returns a list of distance matrices for a polynomial of degree `m`.

    :param s: The parameter `s` is the location of sites, which is expected to be an array. It
    represents the locations of the sites for which the distance matrix is being constructed
    :type s: Arr
    :param m: The parameter `m` represents the degree of the polynomial. It determines the number of
    distance matrices that will be constructed
    :return: The function `construct_poly_matrix` returns a list of arrays.
    """
    n = len(s)
    D1 = construct_distance_matrix(s)
    D = np.ones((n, n))
    Sig = [D]
    for _ in range(m - 1):
        D = np.multiply(D, D1)
        Sig += [D]
    return Sig


def corr_poly(Y, s, m, oracle, corr_core):
    """
    The function `corr_poly` takes in a signal `Y`, a sparsity level `s`, a maximum degree `m`, an
    oracle function, and a correction core function, and returns a polynomial, the number of iterations,
    and a feasibility indicator.

    :param Y: The parameter `Y` represents the input data, which is a vector or matrix of shape
    (n_samples, n_features). It contains the input variables for which we want to find a polynomial
    correlation
    :param s: The parameter `s` represents the degree of the polynomial. It determines the number of
    coefficients in the polynomial
    :param m: The parameter `m` represents the degree of the polynomial that you want to construct. It
    determines the number of coefficients in the polynomial
    :param oracle: The `oracle` parameter is a function that takes in two arguments: `Sig` and `Y`.
    `Sig` is a matrix and `Y` is a vector. The `oracle` function returns a vector `omega`
    :param corr_core: The `corr_core` parameter is a function that takes in the following arguments:
    :return: The function `corr_poly` returns a tuple containing three elements:
    1. A polynomial object representing the polynomial fit to the data.
    2. The number of iterations performed during the correction process.
    3. A boolean value indicating whether a feasible solution was found.
    """
    Sig = construct_poly_matrix(s, m)
    omega = oracle(Sig, Y)
    a, num_iters, feasible = corr_core(Y, m, omega)
    pa = np.ascontiguousarray(a[::-1])
    return np.poly1d(pa), num_iters, feasible
