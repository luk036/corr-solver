# -*- coding: utf-8 -*-

from typing import Any, Tuple

import numpy as np
import pytest
from ellalgo.cutting_plane import cutting_plane_optim
from ellalgo.ell import Ell
from ellalgo.ell_typing import OracleOptim

from corr_solver.corr_bspline_oracle import corr_bspline, generate_bspline_info
from corr_solver.corr_oracle import (
    construct_distance_matrix,
    construct_poly_matrix,
    create_2d_isotropic,
    create_2d_sites,
)
from corr_solver.lsq_corr_oracle import lsq_oracle
from corr_solver.mle_corr_oracle import mle_oracle

# from pytest import approx


site = create_2d_sites(5, 4)
Y = create_2d_isotropic(site, 3000)


def lsq_corr_core2(
    Y: np.ndarray, n: int, omega: OracleOptim[np.ndarray]
) -> Tuple[np.ndarray, int, bool]:
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
    if xbest is None:
        return np.zeros(n), num_iters, False
    return xbest[:-1], num_iters, True


def mle_corr_core(
    Y: np.ndarray, n: int, omega: OracleOptim[np.ndarray]
) -> Tuple[Any, int, bool]:
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


def mle_corr_bspline(Y: np.ndarray, site: np.ndarray, n: int) -> Tuple[Any, int, bool]:
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


def lsq_corr_bspline2(Y: np.ndarray, site: np.ndarray, n: int) -> Tuple[Any, int, bool]:
    """[summary]

    Arguments:
        Y ([type]): [description]
        site ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    return corr_bspline(Y, site, n, lsq_oracle, lsq_corr_core2)


# def test_data() -> None:
#     """[summary]
#     """
#     # assert Y[2,3] == approx(1.9365965488224368)
#     assert site[6, 0] == approx(3.75)
#     # D1 = construct_distance_matrix(site)
#     # assert D1[2, 4] == approx(5.0)


def test_lsq_corr_bspline2() -> None:
    _, num_iters, feasible = lsq_corr_bspline2(Y, site, 4)
    assert feasible
    assert num_iters <= 1054


def test_mle_corr_bspline() -> None:
    _, num_iters, feasible = mle_corr_bspline(Y, site, 4)
    assert feasible
    assert num_iters <= 388


def _design_condition(basis: Any) -> float:
    mat = np.column_stack([np.asarray(B, float).ravel() for B in basis])
    return float(np.linalg.cond(mat))


def test_bspline_basis_is_well_conditioned() -> None:
    for m in (4, 5, 6, 8):
        assert _design_condition(generate_bspline_info(site, m)[0]) < 100.0


def test_bspline_conditions_better_than_poly() -> None:
    m = 8
    bspline = _design_condition(generate_bspline_info(site, m)[0])
    poly = _design_condition(construct_poly_matrix(site, m))
    assert bspline < poly


def test_bspline_fit_is_monotone_decaying() -> None:
    dmax = float(construct_distance_matrix(site).max())
    grid = np.linspace(0.0, dmax, 400)
    for m in (4, 5, 6):
        fitted, _, feasible = corr_bspline(Y, site, m, lsq_oracle, lsq_corr_core2)
        assert feasible
        assert np.all(np.diff(fitted(grid)) <= 1e-9)


def test_bspline_rejects_too_few_control_points() -> None:
    with pytest.raises(ValueError):
        generate_bspline_info(site, 2)
