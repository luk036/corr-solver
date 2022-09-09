# -*- coding: utf-8 -*-
import numpy as np
from ellalgo.cutting_plane import bsearch, bsearch_adaptor, cutting_plane_optim
from ellalgo.ell import Ell
from pytest import approx

from corr_solver.corr_oracle import corr_poly, create_2d_isotropic, create_2d_sites
from corr_solver.lsq_corr_oracle import lsq_oracle
from corr_solver.mle_corr_oracle import mle_oracle
from corr_solver.qmi_oracle import QMIOracle

s = create_2d_sites(5, 4)
Y = create_2d_isotropic(s, 3000)


def lsq_corr_core2(Y, n, P):
    """[summary]

    Arguments:
        Y ([type]): [description]
        n ([type]): [description]
        P ([type]): [description]

    Returns:
        [type]: [description]
    """
    normY = np.linalg.norm(Y, "fro")
    normY2 = 32 * normY * normY
    val = 256 * np.ones(n + 1)
    val[-1] = normY2 * normY2
    x = np.zeros(n + 1)  # cannot all zeros
    x[0] = 1.0
    x[-1] = normY2 / 2
    E = Ell(val, x)
    xb, _, num_iters, _ = cutting_plane_optim(P, E, float("inf"))
    return xb[:-1], num_iters, xb is not None


def lsq_corr_poly2(Y, s, n):
    """[summary]

    Arguments:
        Y ([type]): [description]
        s ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    return corr_poly(Y, s, n, lsq_oracle, lsq_corr_core2)


def lsq_corr_core(Y, n, Q):
    x = np.zeros(n)  # cannot all zeros
    x[0] = 1.0
    E = Ell(256.0, x)
    P = bsearch_adaptor(Q, E)
    normY = np.linalg.norm(Y, "fro")
    upper = normY * normY
    t, num_iters, _ = bsearch(P, [0.0, upper])
    return P.x_best, num_iters, t != upper


def lsq_corr_poly(Y, s, n):
    """[summary]

    Arguments:
        Y ([type]): [description]
        s ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    return corr_poly(Y, s, n, QMIOracle, lsq_corr_core)


def mle_corr_core(Y, n, P):
    """[summary]

    Arguments:
        Y ([type]): [description]
        n ([type]): [description]
        P ([type]): [description]

    Returns:
        [type]: [description]
    """
    x = np.zeros(n)
    x[0] = 1.0
    E = Ell(50.0, x)
    # E.use_parallel_cut = False
    # options = Options()
    # options.max_iter = 2000
    # options.tol = 1e-8
    xb, _, num_iters, _ = cutting_plane_optim(P, E, float("inf"))
    # print(num_iters, feasible, status)
    return xb, num_iters, xb is not None


def mle_corr_poly(Y, s, n):
    """[summary]

    Arguments:
        Y ([type]): [description]
        s ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    _ = np.linalg.cholesky(Y)  # test if Y is SPD.
    return corr_poly(Y, s, n, mle_oracle, mle_corr_core)


def test_data():
    """[summary]"""
    # assert Y[2,3] == approx(1.9365965488224368)
    assert s[6, 0] == approx(8.75)
    # D1 = construct_distance_matrix(s)
    # assert D1[2, 4] == approx(5.0)


def test_lsq_corr_poly():
    _, num_iters, feasible = lsq_corr_poly(Y, s, 4)
    assert feasible
    assert num_iters <= 38


def test_lsq_corr_poly2():
    _, num_iters, feasible = lsq_corr_poly2(Y, s, 4)
    assert feasible
    assert num_iters <= 594


def test_mle_corr_poly():
    _, num_iters, feasible = mle_corr_poly(Y, s, 4)
    assert feasible
    assert num_iters <= 255
