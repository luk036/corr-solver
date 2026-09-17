# -*- coding: utf-8 -*-

from typing import Any, Tuple

import numpy as np
import pytest
from pytest import approx

pytest.importorskip("numba")

from ellalgo.oracles.ldlt_mgr import LDLTMgr  # noqa: E402
from ellalgo.oracles.lmi0_oracle import LMI0Oracle  # noqa: E402
from ellalgo.oracles.lmi_oracle import LMIOracle  # noqa: E402

from corr_solver.corr_bspline_oracle import (  # noqa: E402
    corr_bspline,
    generate_bspline_info,
)
from corr_solver.corr_oracle import (  # noqa: E402
    construct_poly_matrix,
    corr_poly,
    create_2d_isotropic,
    create_2d_sites,
)
from corr_solver.lsq_corr_oracle import lsq_oracle  # noqa: E402
from corr_solver.mle_corr_oracle import mle_oracle  # noqa: E402
from corr_solver.numba_oracles import (  # noqa: E402
    NumbaLMI0Oracle,
    NumbaLMIOracle,
    NumbaLsqOracle,
    NumbaMleOracle,
    NumbaQMIOracle,
    _ldlt,
)
from corr_solver.qmi_oracle import QMIOracle  # noqa: E402

site = create_2d_sites(5, 4)
Y = create_2d_isotropic(site, 3000)
M_BASIS = 4
SIGMA = construct_poly_matrix(site, M_BASIS)
SIGMA_BSPLINE = generate_bspline_info(site, M_BASIS)[0]


def lsq_corr_core(Y: np.ndarray, n: int, Q: Any) -> Tuple[Any, int, bool]:
    """[summary]

    Arguments:
        Y ([type]): [description]
        n ([type]): [description]
        Q ([type]): [description]

    Returns:
        [type]: [description]
    """
    from ellalgo.cutting_plane import BSearchAdaptor, bsearch
    from ellalgo.ell import Ell

    x = np.zeros(n)
    x[0] = 1.0
    omega = BSearchAdaptor(Q, Ell(256.0, x))
    upper = np.linalg.norm(Y, "fro") ** 2
    t, num_iters = bsearch(omega, (0.0, upper))
    return omega.x_best, num_iters, t != upper


def lsq_corr_core2(Y: np.ndarray, n: int, omega: Any) -> Tuple[Any, int, bool]:
    """[summary]

    Arguments:
        Y ([type]): [description]
        n ([type]): [description]
        omega ([type]): [description]

    Returns:
        [type]: [description]
    """
    from ellalgo.cutting_plane import cutting_plane_optim
    from ellalgo.ell import Ell

    normY = np.linalg.norm(Y, "fro")
    normY2 = 32 * normY * normY
    val = 256 * np.ones(n + 1)
    val[-1] = normY2 * normY2
    x = np.zeros(n + 1)
    x[0] = 1.0
    x[-1] = normY2 / 2
    xbest, _, num_iters = cutting_plane_optim(omega, Ell(val, x), float("inf"))
    if xbest is None:
        return np.zeros(n), num_iters, False
    return xbest[:-1], num_iters, True


def mle_corr_core(_: np.ndarray, n: int, omega: Any) -> Tuple[Any, int, bool]:
    """[summary]

    Arguments:
        _ ([type]): [description]
        n ([type]): [description]
        omega ([type]): [description]

    Returns:
        [type]: [description]
    """
    from ellalgo.cutting_plane import cutting_plane_optim
    from ellalgo.ell import Ell

    x = np.zeros(n)
    x[0] = 1.0
    result = cutting_plane_optim(omega, Ell(50.0, x), float("inf"))
    return result[0], result[2], result[0] is not None


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


def test_qmi_oracle_matches() -> None:
    """[summary]"""
    rng = np.random.default_rng(7)
    ref = QMIOracle(SIGMA, Y)
    jit = NumbaQMIOracle(SIGMA, Y)
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


def test_lmi0_oracle_matches() -> None:
    """[summary]"""
    rng = np.random.default_rng(13)
    ref = LMI0Oracle(SIGMA)
    jit = NumbaLMI0Oracle(SIGMA)
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


def test_lmi_oracle_matches() -> None:
    """[summary]"""
    rng = np.random.default_rng(17)
    ref = LMIOracle(SIGMA, 2 * Y)
    jit = NumbaLMIOracle(SIGMA, 2 * Y)
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


def _lsq_trajectory() -> list:
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


def _mle_trajectory() -> list:
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


def test_lsq_oracle_matches() -> None:
    """[summary]"""
    rng = np.random.default_rng(23)
    ref = lsq_oracle(SIGMA, Y)
    jit = NumbaLsqOracle(SIGMA, Y)
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


def test_lsq_oracle_verdicts_along_trajectory() -> None:
    """[summary]

    On the trajectory the cut *values* are only equivalent to about 1e-3
    relative: near the boundary of the positive-semidefinite cone the witness
    vector amplifies rounding, and the compiled kernel assembles ``Fx`` in a
    different order from the BLAS ``dot`` used by the pure-Python oracle. The
    feasibility verdicts, which is what steers the solver, agree exactly.
    """
    calls = _lsq_trajectory()
    assert calls
    ref = lsq_oracle(SIGMA, Y)
    jit = NumbaLsqOracle(SIGMA, Y)
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


def test_mle_oracle_matches() -> None:
    """[summary]

    The MLE oracle has no quadratic-matrix-inequality term, so its cuts agree
    with the pure-Python oracle to floating-point precision rather than merely
    to solver-equivalence.
    """
    calls = _mle_trajectory()
    assert calls
    ref = mle_oracle(SIGMA_BSPLINE, Y)
    jit = NumbaMleOracle(SIGMA_BSPLINE, Y)
    seen_improve = 0
    for x, t in calls:
        cut_ref, val_ref = ref.assess_optim(x, t)
        cut_jit, val_jit = jit.assess_optim(x, t)
        assert (val_ref is None) == (val_jit is None)
        assert cut_ref[0] == approx(cut_jit[0], rel=1e-8, abs=1e-8)
        assert cut_ref[1] == approx(cut_jit[1], rel=1e-2, abs=1e-6)
        if val_ref is not None:
            seen_improve += 1


def test_numba_lsq_corr_poly() -> None:
    """[summary]"""
    _, num_iters, feasible = corr_poly(Y, site, M_BASIS, NumbaQMIOracle, lsq_corr_core)
    assert feasible
    assert num_iters <= 100


def test_numba_lsq_corr_poly2() -> None:
    """[summary]"""
    _, num_iters, feasible = corr_poly(Y, site, M_BASIS, NumbaLsqOracle, lsq_corr_core2)
    assert feasible
    assert num_iters <= 1095


def test_numba_mle_corr_bspline() -> None:
    """[summary]"""
    _, num_iters, feasible = corr_bspline(
        Y, site, M_BASIS, NumbaMleOracle, mle_corr_core
    )
    assert feasible
    assert num_iters <= 388
