# -*- coding: utf-8 -*-
import numpy as np
from ellalgo.cutting_plane import BSearchAdaptor, bsearch, cutting_plane_optim
from ellalgo.ell import Ell
from pytest import approx

from corr_solver.corr_oracle import corr_poly, create_2d_isotropic, create_2d_sites
from corr_solver.lsq_corr_oracle import lsq_oracle
from corr_solver.mle_corr_oracle import mle_oracle
from corr_solver.qmi_oracle import QMIOracle

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


def lsq_corr_poly2(Y, site, n):
    """[summary]

    Arguments:
        Y ([type]): [description]
        site ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    return corr_poly(Y, site, n, lsq_oracle, lsq_corr_core2)


def lsq_corr_core(Y, n, Q):
    x = np.zeros(n)  # cannot all zeros
    x[0] = 1.0
    ellip = Ell(256.0, x)
    omega = BSearchAdaptor(Q, ellip)
    normY = np.linalg.norm(Y, "fro")
    upper = normY * normY
    t, num_iters = bsearch(omega, [0.0, upper])
    return omega.x_best, num_iters, t != upper


def lsq_corr_poly(Y, site, n):
    """[summary]

    Arguments:
        Y ([type]): [description]
        site ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    return corr_poly(Y, site, n, QMIOracle, lsq_corr_core)


def mle_corr_core(_, n, omega):
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
    # options = Options()
    # options.max_iters = 2000
    # options.tol = 1e-8
    xbest, _, num_iters = cutting_plane_optim(omega, ellip, float("inf"))
    return xbest, num_iters, xbest is not None


def mle_corr_poly(Y, site, n):
    """[summary]

    Arguments:
        Y ([type]): [description]
        site ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    _ = np.linalg.cholesky(Y)  # test if Y is SPD.
    return corr_poly(Y, site, n, mle_oracle, mle_corr_core)


def test_data():
    """[summary]"""
    assert site[6, 0] == approx(8.75)


def test_lsq_corr_poly():
    _, num_iters, feasible = lsq_corr_poly(Y, site, 4)
    assert feasible
    assert num_iters <= 2000


def test_lsq_corr_poly2():
    _, num_iters, feasible = lsq_corr_poly2(Y, site, 4)
    assert feasible
    assert num_iters <= 1076


# def test_mle_corr_poly():
#     _, num_iters, feasible = mle_corr_poly(Y, site, 4)
#     assert feasible
#     assert num_iters <= 255
