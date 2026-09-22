# -*- coding: utf-8 -*-

from typing import Any

import numpy as np
import pytest
from pytest import approx

pytest.importorskip("numba")

from ellalgo.oracles.ldlt_mgr import LDLTMgr  # noqa: E402

from corr_solver.backends import make_backend  # noqa: E402
from corr_solver.basis import generate_bspline_info  # noqa: E402
from corr_solver.corr_bspline_oracle import corr_bspline  # noqa: E402
from corr_solver.corr_oracle import construct_poly_matrix, corr_poly  # noqa: E402
from corr_solver.numba_oracles import (  # noqa: E402
    NumbaLsqOracle,
    NumbaMleOracle,
    NumbaQMIOracle,
    _ldlt,
)
from corr_solver.solvers import (  # noqa: E402
    lsq_corr_core,
    lsq_corr_core2,
    mle_corr_core,
)

M_BASIS = 4


@pytest.fixture(scope="module")
def sigma(site: np.ndarray) -> list:
    """[summary]"""
    return construct_poly_matrix(site, M_BASIS)


@pytest.fixture(scope="module")
def sigma_bspline(site: np.ndarray) -> list:
    """[summary]"""
    return generate_bspline_info(site, M_BASIS)[0]


def test_ldlt_matches_ldlt_mgr() -> None:
    """[summary]"""
    rng = np.random.default_rng(11)
    n = 20
    seen_indefinite = 0
    for _ in range(40):
        b = rng.standard_normal((n, n))
        mat = (b + b.T) / 2
        mat = np.ascontiguousarray(mat @ mat.T - rng.uniform(0.0, 3.0) * np.eye(n))

        mgr = LDLTMgr(n)
        spd = mgr.factorize(mat)
        lmat = np.zeros((n, n))
        diag = np.zeros(n)
        wit = np.zeros(n)
        pos1 = _ldlt(mat, n, lmat, diag, wit)

        if spd:
            assert pos1 == 0
            continue

        seen_indefinite += 1
        ep = mgr.witness()
        assert pos1 == mgr.pos[1]
        assert -diag[pos1 - 1] == approx(ep, rel=1e-9)
        assert wit[:pos1] == approx(mgr.wit[:pos1], rel=1e-9)
        assert mgr.sym_quad(mat) == approx(
            float(wit[:pos1] @ mat[:pos1, :pos1] @ wit[:pos1]), rel=1e-9
        )
    assert seen_indefinite > 0


def test_qmi_oracle_matches(sigma: list, Y: np.ndarray) -> None:
    """[summary]"""
    rng = np.random.default_rng(7)
    ref = make_backend("python").qmi(sigma, Y)
    jit = make_backend("numba").qmi(sigma, Y)
    seen_infeasible = 0
    for _ in range(40):
        x = rng.standard_normal(M_BASIS)
        ref.update(50.0)
        jit.update(50.0)
        cut_ref = ref.assess_feas(x)
        cut_jit = jit.assess_feas(x)
        assert (cut_ref is None) == (cut_jit is None)
        if cut_ref is not None:
            assert cut_jit is not None
            seen_infeasible += 1
            assert cut_ref[0] == approx(cut_jit[0], rel=1e-8)
            assert cut_ref[1] == approx(cut_jit[1], rel=1e-8)
    assert seen_infeasible > 0


def test_lmi0_oracle_matches(sigma: list) -> None:
    """[summary]"""
    rng = np.random.default_rng(13)
    ref = make_backend("python").lmi0(sigma)
    jit = make_backend("numba").lmi0(sigma)
    seen_infeasible = 0
    for _ in range(40):
        x = rng.standard_normal(M_BASIS)
        cut_ref = ref.assess_feas(x)
        cut_jit = jit.assess_feas(x)
        assert (cut_ref is None) == (cut_jit is None)
        if cut_ref is not None:
            assert cut_jit is not None
            seen_infeasible += 1
            assert cut_ref[0] == approx(cut_jit[0], rel=1e-8)
            assert cut_ref[1] == approx(cut_jit[1], rel=1e-8)
    assert seen_infeasible > 0


def test_lmi_oracle_matches(sigma: list, Y: np.ndarray) -> None:
    """[summary]"""
    rng = np.random.default_rng(17)
    ref = make_backend("python").lmi(sigma, 2 * Y)
    jit = make_backend("numba").lmi(sigma, 2 * Y)
    seen_infeasible = 0
    for _ in range(40):
        x = rng.standard_normal(M_BASIS)
        cut_ref = ref.assess_feas(x)
        cut_jit = jit.assess_feas(x)
        assert (cut_ref is None) == (cut_jit is None)
        if cut_ref is not None:
            assert cut_jit is not None
            seen_infeasible += 1
            assert cut_ref[0] == approx(cut_jit[0], rel=1e-8)
            assert cut_ref[1] == approx(cut_jit[1], rel=1e-8)
    assert seen_infeasible > 0


def _lsq_trajectory(Y: np.ndarray, site: np.ndarray) -> list:
    """[summary]

    Returns:
        [type]: [description]
    """
    calls: list = []

    class Recorder(NumbaLsqOracle):
        def assess_optim(self, x: np.ndarray, t: float) -> Any:
            calls.append((np.array(x), t))
            return super().assess_optim(x, t)

    corr_poly(Y, site, M_BASIS, lambda F, F0: Recorder(F, F0), lsq_corr_core2)
    return calls


def _mle_trajectory(Y: np.ndarray, site: np.ndarray) -> list:
    """[summary]

    Returns:
        [type]: [description]
    """
    calls: list = []

    class Recorder(NumbaMleOracle):
        def assess_optim(self, x: np.ndarray, t: float) -> Any:
            calls.append((np.array(x), t))
            return super().assess_optim(x, t)

    corr_bspline(Y, site, M_BASIS, lambda F, F0: Recorder(F, F0), mle_corr_core)
    return calls


def test_lsq_oracle_matches(sigma: list, Y: np.ndarray) -> None:
    """[summary]"""
    rng = np.random.default_rng(23)
    ref = make_backend("python").lsq(sigma, Y)
    jit = make_backend("numba").lsq(sigma, Y)
    seen_infeasible = 0
    for _ in range(40):
        x = np.zeros(M_BASIS + 1)
        x[:-1] = rng.standard_normal(M_BASIS)
        x[-1] = rng.uniform(0.0, 100.0)
        t = rng.uniform(0.0, 100.0)
        cut_ref, val_ref = ref.assess_optim(x, t)
        cut_jit, val_jit = jit.assess_optim(x, t)
        assert (val_ref is None) == (val_jit is None)
        assert cut_ref[0] == approx(cut_jit[0], rel=1e-8)
        assert cut_ref[1] == approx(cut_jit[1], rel=1e-8)
        if val_ref is None:
            seen_infeasible += 1
    assert seen_infeasible > 0


def test_lsq_oracle_verdicts_along_trajectory(
    site: np.ndarray, Y: np.ndarray, sigma: list
) -> None:
    """[summary]

    On the trajectory the cut *values* are only equivalent to about 1e-3
    relative: near the boundary of the positive-semidefinite cone the witness
    vector amplifies rounding, and the compiled kernel assembles ``Fx`` in a
    different order from the BLAS ``dot`` used by the pure-Python oracle. The
    feasibility verdicts, which is what steers the solver, agree exactly.
    """
    calls = _lsq_trajectory(Y, site)
    assert calls
    ref = make_backend("python").lsq(sigma, Y)
    jit = make_backend("numba").lsq(sigma, Y)
    seen_infeasible = 0
    seen_improve = 0
    for x, t in calls:
        cut_ref, val_ref = ref.assess_optim(x, t)
        cut_jit, val_jit = jit.assess_optim(x, t)
        assert (val_ref is None) == (val_jit is None)
        if val_ref is not None:
            seen_improve += 1
        else:
            seen_infeasible += 1
    assert seen_infeasible > 0
    assert seen_improve > 0


def test_mle_oracle_matches(
    site: np.ndarray, Y: np.ndarray, sigma_bspline: list
) -> None:
    """[summary]

    The MLE oracle has no quadratic-matrix-inequality term, so its cuts agree
    with the pure-Python oracle to floating-point precision rather than merely
    to solver-equivalence.
    """
    calls = _mle_trajectory(Y, site)
    assert calls
    ref = make_backend("python").mle(sigma_bspline, Y)
    jit = make_backend("numba").mle(sigma_bspline, Y)
    seen_improve = 0
    for x, t in calls:
        cut_ref, val_ref = ref.assess_optim(x, t)
        cut_jit, val_jit = jit.assess_optim(x, t)
        assert (val_ref is None) == (val_jit is None)
        assert cut_ref[0] == approx(cut_jit[0], rel=1e-8, abs=1e-8)
        assert cut_ref[1] == approx(cut_jit[1], rel=1e-2, abs=1e-6)
        if val_ref is not None:
            seen_improve += 1

    assert seen_improve > 0


def test_make_backend_rejects_unknown() -> None:
    """[summary]"""
    with pytest.raises(ValueError):
        make_backend("nope")


def test_numba_lsq_corr_poly(site: np.ndarray, Y: np.ndarray) -> None:
    """[summary]"""
    _, num_iters, feasible = corr_poly(Y, site, M_BASIS, NumbaQMIOracle, lsq_corr_core)
    assert feasible
    assert num_iters <= 100


def test_numba_lsq_corr_poly2(site: np.ndarray, Y: np.ndarray) -> None:
    """[summary]"""
    _, num_iters, feasible = corr_poly(Y, site, M_BASIS, NumbaLsqOracle, lsq_corr_core2)
    assert feasible
    assert num_iters <= 1095


def test_numba_mle_corr_bspline(site: np.ndarray, Y: np.ndarray) -> None:
    """[summary]"""
    _, num_iters, feasible = corr_bspline(
        Y, site, M_BASIS, NumbaMleOracle, mle_corr_core
    )
    assert feasible
    assert num_iters <= 388
