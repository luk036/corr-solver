# -*- coding: utf-8 -*-

from typing import Any

import numpy as np
import pytest

from corr_solver.corr_bspline_oracle import corr_bspline, generate_bspline_info
from corr_solver.corr_oracle import construct_distance_matrix, construct_poly_matrix
from corr_solver.lsq_corr_oracle import lsq_oracle
from corr_solver.mle_corr_oracle import mle_oracle
from corr_solver.solvers import lsq_corr_core2, mle_corr_core


def mle_corr_bspline(Y: np.ndarray, site: np.ndarray, n: int):
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


def lsq_corr_bspline2(Y: np.ndarray, site: np.ndarray, n: int):
    """[summary]

    Arguments:
        Y ([type]): [description]
        site ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    return corr_bspline(Y, site, n, lsq_oracle, lsq_corr_core2)


def _design_condition(basis: Any) -> float:
    mat = np.column_stack([np.asarray(B, float).ravel() for B in basis])
    return float(np.linalg.cond(mat))


def test_lsq_corr_bspline2(site: np.ndarray, Y: np.ndarray) -> None:
    _, num_iters, feasible = lsq_corr_bspline2(Y, site, 4)
    assert feasible
    assert num_iters <= 1054


def test_mle_corr_bspline(site: np.ndarray, Y: np.ndarray) -> None:
    _, num_iters, feasible = mle_corr_bspline(Y, site, 4)
    assert feasible
    assert num_iters <= 388


def test_bspline_basis_is_well_conditioned(site: np.ndarray) -> None:
    for m in (4, 5, 6, 8):
        assert _design_condition(generate_bspline_info(site, m)[0]) < 100.0


def test_bspline_conditions_better_than_poly(site: np.ndarray) -> None:
    m = 8
    bspline = _design_condition(generate_bspline_info(site, m)[0])
    poly = _design_condition(construct_poly_matrix(site, m))
    assert bspline < poly


def test_bspline_fit_is_monotone_decaying(site: np.ndarray, Y: np.ndarray) -> None:
    dmax = float(construct_distance_matrix(site).max())
    grid = np.linspace(0.0, dmax, 400)
    for m in (4, 5, 6):
        fitted, _, feasible = corr_bspline(Y, site, m, lsq_oracle, lsq_corr_core2)
        assert feasible
        assert np.all(np.diff(fitted(grid)) <= 1e-9)


def test_bspline_rejects_too_few_control_points(site: np.ndarray) -> None:
    with pytest.raises(ValueError):
        generate_bspline_info(site, 2)
