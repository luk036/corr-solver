# -*- coding: utf-8 -*-

import numpy as np
from pytest import approx

from corr_solver.corr_oracle import corr_poly
from corr_solver.lsq_corr_oracle import lsq_oracle
from corr_solver.mle_corr_oracle import mle_oracle
from corr_solver.qmi_oracle import QMIOracle
from corr_solver.solvers import lsq_corr_core, lsq_corr_core2, mle_corr_core


def lsq_corr_poly2(Y: np.ndarray, site: np.ndarray, n: int):
    """[summary]

    Arguments:
        Y ([type]): [description]
        site ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    return corr_poly(Y, site, n, lsq_oracle, lsq_corr_core2)


def lsq_corr_poly(Y: np.ndarray, site: np.ndarray, n: int):
    """[summary]

    Arguments:
        Y ([type]): [description]
        site ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    return corr_poly(Y, site, n, QMIOracle, lsq_corr_core)


def mle_corr_poly(Y: np.ndarray, site: np.ndarray, n: int):
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


def test_data(site: np.ndarray) -> None:
    """[summary]"""
    assert site[6, 0] == approx(8.75)


def test_lsq_corr_poly(site: np.ndarray, Y: np.ndarray) -> None:
    _, num_iters, feasible = lsq_corr_poly(Y, site, 4)
    assert feasible
    # A bisection at float precision needs ~62 steps here. The bound is tight
    # enough to catch a regression in the search-space termination: without the
    # floating-point stall guard in ellalgo's `bsearch`, this runs to the
    # 2000-iteration cap because `tau < tolerance` is unreachable.
    assert num_iters <= 100


def test_lsq_corr_poly2(site: np.ndarray, Y: np.ndarray) -> None:
    _, num_iters, feasible = lsq_corr_poly2(Y, site, 4)
    assert feasible
    assert num_iters <= 1095


# def test_mle_corr_poly(site: np.ndarray, Y: np.ndarray) -> None:
#     _, num_iters, feasible = mle_corr_poly(Y, site, 4)
#     assert feasible
#     assert num_iters <= 255
