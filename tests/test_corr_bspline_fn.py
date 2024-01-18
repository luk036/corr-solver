# -*- coding: utf-8 -*-
from pytest import approx

import numpy as np

from ellalgo.cutting_plane import bsearch, BSearchAdaptor, cutting_plane_optim
from ellalgo.ell import Ell
from corr_solver.corr_oracle import create_2d_isotropic, create_2d_sites
from corr_solver.corr_bspline_oracle import corr_bspline
from corr_solver.lsq_corr_oracle import lsq_oracle
from corr_solver.mle_corr_oracle import mle_oracle

site = create_2d_sites(5, 4)
Y = create_2d_isotropic(site, 3000)


def lsq_corr_core2(Y, n, omega):
    """[summary]

    Arguments:
        Y ([type]): [description]
        n ([type]): [description]
        omega ([type]): [description]

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
    ellip = Ell(val, x)
    xbest, _, num_iters = cutting_plane_optim(omega, ellip, float("inf"))
    return xbest[:-1], num_iters, xbest is not None


def mle_corr_core(Y, n, omega):
    """[summary]

    Arguments:
        Y ([type]): [description]
        n ([type]): [description]
        omega ([type]): [description]

    Returns:
        [type]: [description]
    """
    x = np.zeros(n)
    x[0] = 1.0
    ellip = Ell(50.0, x)
    # ellip.use_parallel_cut = False
    # options = Options()
    # options.max_iters = 2000
    # options.tol = 1e-8
    xbest, _, num_iters = cutting_plane_optim(omega, ellip, float("inf"))
    # print(num_iters, feasible, status)
    return xbest, num_iters, xbest is not None


def mle_corr_bspline(Y, site, n):
    """[summary]

    Arguments:
        Y ([type]): [description]
        site ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    _ = np.linalg.cholesky(Y)  # test if Y is SPD.
    return corr_bspline(Y, site, n, mle_oracle, mle_corr_core)


def lsq_corr_bspline2(Y, site, n):
    """[summary]

    Arguments:
        Y ([type]): [description]
        site ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    return corr_bspline(Y, site, n, lsq_oracle, lsq_corr_core2)


# def test_data():
#     """[summary]
#     """
#     # assert Y[2,3] == approx(1.9365965488224368)
#     assert site[6, 0] == approx(3.75)
#     # D1 = construct_distance_matrix(site)
#     # assert D1[2, 4] == approx(5.0)


def test_lsq_corr_bspline2():
    _, num_iters, feasible = lsq_corr_bspline2(Y, site, 4)
    assert feasible
    assert num_iters <= 1054


def test_mle_corr_bspline():
    _, num_iters, feasible = mle_corr_bspline(Y, site, 4)
    assert feasible
    assert num_iters <= 384
