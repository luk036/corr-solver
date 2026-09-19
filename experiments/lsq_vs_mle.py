"""Compare the LSQ and MLE correlation solvers on isotropic and anisotropic problems.

Isotropic data comes from ``create_2d_isotropic``; anisotropic data reuses the
same Gaussian kernel with per-axis widths,
``exp(-0.12 * ((dx/lx)^2 + (dy/ly)^2))``. Both solver families fit the same
Euclidean-distance basis (polynomial or B-spline), so the anisotropic runs
expose model misspecification.

The two solvers optimise different problems: LSQ minimises the Frobenius
residual ``||Y - Omega||`` over ``Omega >= 0``, whereas MLE minimises
``log det Omega + Tr(Omega^-1 Y)`` over ``0 <= Omega <= 2Y``. The ``2Y_ok``
column records whether a returned ``Omega`` satisfies MLE's upper bound, which
is why LSQ can show a lower ``mle_obj`` at a point MLE is not allowed to reach.
The correlation figure overlays the original generating covariance (black
markers) and its per-axis kernels (black dashed/dotted lines) on the fits.
Results are printed as a table and saved as PNG figures.
"""

import time
from typing import Any, Dict, List, Tuple

import numpy as np
from ellalgo.cutting_plane import BSearchAdaptor, bsearch, cutting_plane_optim
from ellalgo.ell import Ell

from corr_solver.corr_bspline_oracle import corr_bspline, generate_bspline_info
from corr_solver.corr_oracle import (
    construct_distance_matrix,
    construct_poly_matrix,
    corr_poly,
    create_2d_isotropic,
    create_2d_sites,
)
from corr_solver.lsq_corr_oracle import lsq_oracle
from corr_solver.mle_corr_oracle import mle_oracle
from corr_solver.qmi_oracle import QMIOracle

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

Arr = np.ndarray

SDKERN = 0.12
VAR = 2.0
TAU = 0.00001

PROBLEM_SPECS: Dict[str, Tuple[float, float]] = {
    "iso (1,1)": (1.0, 1.0),
    "aniso (1,3)": (1.0, 3.0),
    "aniso (3,1)": (3.0, 1.0),
}


def lsq_corr_core2(Y: Arr, n: int, omega: Any) -> Tuple[Arr, int, bool]:
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


def lsq_corr_core(Y: Arr, n: int, Q: Any) -> Tuple[Arr, int, bool]:
    x = np.zeros(n)
    x[0] = 1.0
    omega = BSearchAdaptor(Q, Ell(256.0, x))
    upper = np.linalg.norm(Y, "fro") ** 2
    t, num_iters = bsearch(omega, (0.0, upper))
    return omega.x_best, num_iters, t != upper


def mle_corr_core(_: Arr, n: int, omega: Any) -> Tuple[Arr, int, bool]:
    x = np.zeros(n)
    x[0] = 1.0
    xbest, _, num_iters = cutting_plane_optim(omega, Ell(50.0, x), float("inf"))
    return xbest, num_iters, xbest is not None


def create_2d_anisotropic(
    site: Arr, length_x: float, length_y: float, N: int = 3000
) -> Arr:
    n = site.shape[0]
    rng = np.random.RandomState(5)

    dx = site[:, None, 0] - site[None, :, 0]
    dy = site[:, None, 1] - site[None, :, 1]
    dist_sq = (dx / length_x) ** 2 + (dy / length_y) ** 2
    Sigma = np.exp(-SDKERN * dist_sq)

    A = np.linalg.cholesky(Sigma)
    Y = np.zeros((n, n))
    outer_buf = np.empty((n, n))
    for _ in range(N):
        y = A @ (VAR * rng.randn(n)) + TAU * rng.randn(n)
        np.outer(y, y, out=outer_buf)
        Y += outer_buf
    return Y / N


def true_covariance(site: Arr, length_x: float, length_y: float) -> Arr:
    dx = site[:, None, 0] - site[None, :, 0]
    dy = site[:, None, 1] - site[None, :, 1]
    dist_sq = (dx / length_x) ** 2 + (dy / length_y) ** 2
    C = (VAR**2) * np.exp(-SDKERN * dist_sq)
    np.fill_diagonal(C, C.diagonal() + TAU**2)
    return C


def true_kernel(h: Arr, length: float) -> Arr:
    return (VAR**2) * np.exp(-SDKERN * (h / length) ** 2)


def mle_objective(omega: Arr, Y: Arr) -> float:
    sign, logdet = np.linalg.slogdet(omega)
    if sign <= 0:
        return float("nan")
    return float(logdet + np.trace(np.linalg.solve(omega, Y)))


SOLVERS: Dict[str, Dict[str, Any]] = {
    "LSQ-QMI": {"basis": "poly", "oracle": QMIOracle, "core": lsq_corr_core},
    "LSQ-opt": {"basis": "poly", "oracle": lsq_oracle, "core": lsq_corr_core2},
    "MLE": {"basis": "poly", "oracle": mle_oracle, "core": mle_corr_core},
    "LSQ-opt-bspl": {
        "basis": "bspline",
        "oracle": lsq_oracle,
        "core": lsq_corr_core2,
    },
    "MLE-bspl": {"basis": "bspline", "oracle": mle_oracle, "core": mle_corr_core},
}

POLY_SOLVERS = ["LSQ-QMI", "LSQ-opt", "MLE"]


def build_problems(site: Arr, N: int = 3000) -> Dict[str, Arr]:
    out: Dict[str, Arr] = {}
    for name, (length_x, length_y) in PROBLEM_SPECS.items():
        if length_x == length_y:
            out[name] = create_2d_isotropic(site, N)
        else:
            out[name] = create_2d_anisotropic(site, length_x, length_y, N)
    return out


def run_solver(
    Y: Arr, site: Arr, m: int, name: str, spec: Dict[str, Any]
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if spec["basis"] == "poly":
        fitted, iters, feasible = corr_poly(Y, site, m, spec["oracle"], spec["core"])
        basis = construct_poly_matrix(site, m)
        coeffs = np.asarray(fitted.c)[::-1]
    else:
        fitted, iters, feasible = corr_bspline(Y, site, m, spec["oracle"], spec["core"])
        basis, _, _ = generate_bspline_info(site, m)
        coeffs = np.asarray(fitted.c)
    elapsed = time.perf_counter() - t0

    omega = np.zeros_like(Y)
    if feasible and len(coeffs) == len(basis):
        for c, fk in zip(coeffs, basis):
            omega += c * fk

    normY = float(np.linalg.norm(Y, "fro"))
    resid = float(np.linalg.norm(Y - omega, "fro"))
    return {
        "solver": name,
        "basis": spec["basis"],
        "feasible": bool(feasible),
        "iters": int(iters),
        "time": elapsed,
        "rel_err": resid / normY,
        "resid": resid,
        "mle_obj": mle_objective(omega, Y),
        "mle_constr": bool(np.linalg.eigvalsh(2 * Y - omega).min() >= -1e-6),
        "fitted": fitted,
    }


def format_table(rows: List[Dict[str, Any]]) -> str:
    header = (
        f"{'solver':14s} {'basis':8s} {'problem':12s} {'feas':5s} "
        f"{'iters':>6s} {'rel_err':>10s} {'mle_obj':>10s} {'2Y_ok':>6s} "
        f"{'time(s)':>8s}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['solver']:14s} {r['basis']:8s} {r['problem']:12s} "
            f"{str(r['feasible']):5s} {r['iters']:6d} {r['rel_err']:10.4e} "
            f"{r['mle_obj']:10.3f} {str(r['mle_constr']):>6s} {r['time']:8.2f}"
        )
    return "\n".join(lines)


def plot_bars(rows: List[Dict[str, Any]], path: str) -> None:
    solvers = list(dict.fromkeys(r["solver"] for r in rows))
    problems = list(dict.fromkeys(r["problem"] for r in rows))
    x = np.arange(len(problems), dtype=float)
    width = 0.8 / len(solvers)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    metrics = [
        ("rel_err", "Relative fit error", True),
        ("iters", "Iterations", False),
        ("time", "Runtime (s)", False),
    ]
    for ax, (key, title, logy) in zip(axes, metrics):
        for k, solver in enumerate(solvers):
            vals = [
                next(
                    r[key] for r in rows if r["solver"] == solver and r["problem"] == p
                )
                for p in problems
            ]
            ax.bar(x + k * width, vals, width, label=solver)
        ax.set_xticks(x + width * (len(solvers) - 1) / 2)
        ax.set_xticklabels(problems, rotation=15, ha="right", fontsize=8)
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_correlations(
    rows: List[Dict[str, Any]], problems: Dict[str, Arr], site: Arr, path: str
) -> None:
    D = construct_distance_matrix(site)
    iu = np.triu_indices_from(D)
    ds = D[iu]

    fig, axes = plt.subplots(1, len(problems), figsize=(6 * len(problems), 4.5))
    for ax, (pname, Y) in zip(np.atleast_1d(axes), problems.items()):
        length_x, length_y = PROBLEM_SPECS[pname]
        C_true = true_covariance(site, length_x, length_y)
        grid = np.linspace(0.0, float(ds.max()), 200)
        ax.scatter(ds, Y[iu], s=6, c="0.6", alpha=0.45, label="empirical")
        ax.scatter(
            ds,
            C_true[iu],
            s=10,
            c="k",
            marker="x",
            alpha=0.45,
            label="true (generating)",
        )
        ax.plot(grid, true_kernel(grid, length_x), "k--", lw=1.4, label="true kernel x")
        if length_y != length_x:
            ax.plot(
                grid, true_kernel(grid, length_y), "k:", lw=1.4, label="true kernel y"
            )
        for solver in POLY_SOLVERS:
            r = next(r for r in rows if r["solver"] == solver and r["problem"] == pname)
            ax.plot(grid, r["fitted"](grid), lw=1.7, label=solver)
        ax.set_title(pname)
        ax.set_xlabel("Distance h")
        ax.set_ylabel("Covariance")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    site = create_2d_sites(5, 4)
    m = 4
    problems = build_problems(site)

    for pname, Y in problems.items():
        print(f"{pname:12s}  n={Y.shape[0]}  cond={np.linalg.cond(Y):.1f}")

    rows: List[Dict[str, Any]] = []
    for pname, Y in problems.items():
        for sname, spec in SOLVERS.items():
            r = run_solver(Y, site, m, sname, spec)
            r["problem"] = pname
            rows.append(r)

    print("\n" + format_table(rows))

    if HAS_MPL:
        plot_bars(rows, "lsq_vs_mle_metrics.png")
        plot_correlations(rows, problems, site, "lsq_vs_mle_corr.png")
        print("\nsaved lsq_vs_mle_metrics.png, lsq_vs_mle_corr.png")


if __name__ == "__main__":
    main()
