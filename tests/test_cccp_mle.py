# -*- coding: utf-8 -*-

import numpy as np
from ellalgo.cutting_plane import cutting_plane_optim
from ellalgo.ell import Ell
from scipy.interpolate import BSpline

from corr_solver.cccp_mle_oracle import cccp_mle, mle_obj, omega_of
from corr_solver.corr_bspline_oracle import (
    generate_bspline_info,
    mono_decreasing_oracle2,
)
from corr_solver.corr_oracle import (
    construct_distance_matrix,
    construct_poly_matrix,
    create_2d_isotropic,
    create_2d_sites,
)
from corr_solver.lsq_corr_oracle import lsq_oracle

site = create_2d_sites(5, 4)
Y = create_2d_isotropic(site, 3000)
M = 4


def lsq_start(Sigma: list) -> np.ndarray:
    n = len(Sigma)
    normY = np.linalg.norm(Y, "fro")
    normY2 = 32 * normY * normY
    val = 256 * np.ones(n + 1)
    val[-1] = normY2 * normY2
    x = np.zeros(n + 1)
    x[0] = 1.0
    x[-1] = normY2 / 2
    xbest, _, _ = cutting_plane_optim(lsq_oracle(Sigma, Y), Ell(val, x), float("inf"))
    return xbest[:-1]


def test_cccp_mle_poly_reduces_objective() -> None:
    Sigma = construct_poly_matrix(site, M)
    x0 = lsq_start(Sigma)
    x, n_outer = cccp_mle(Y, Sigma, x0)
    assert n_outer >= 1
    assert mle_obj(x, Sigma, Y) <= mle_obj(x0, Sigma, Y)
    assert np.linalg.eigvalsh(omega_of(x, Sigma)).min() >= -1e-8


def test_cccp_mle_bspline_is_monotone() -> None:
    Sigma, t, k = generate_bspline_info(site, M)
    x0 = lsq_start(Sigma)
    x, n_outer = cccp_mle(
        Y, Sigma, x0, wrapper=lambda oracle: mono_decreasing_oracle2(oracle, M)
    )
    assert n_outer >= 1
    assert np.all(np.diff(x) <= 1e-9)
    grid = np.linspace(0.0, float(construct_distance_matrix(site).max()), 400)
    assert np.all(np.diff(BSpline(t, x, k)(grid)) <= 1e-9)
    assert mle_obj(x, Sigma, Y) <= mle_obj(x0, Sigma, Y)
    assert np.linalg.eigvalsh(omega_of(x, Sigma)).min() >= -1e-8


def test_cccp_mle_respects_n_outer() -> None:
    Sigma = construct_poly_matrix(site, M)
    x0 = lsq_start(Sigma)
    _, n_outer = cccp_mle(Y, Sigma, x0, n_outer=1)
    assert n_outer == 1
