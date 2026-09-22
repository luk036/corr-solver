"""Generate the SVG figures for the LSQ-vs-MLE slide deck."""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ellalgo.cutting_plane import cutting_plane_optim  # noqa: E402
from ellalgo.ell import Ell  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    PROBLEM_SPECS,
    build_problems,
    true_covariance,
    true_kernel,
)
from lsq_vs_mle import POLY_SOLVERS, SOLVERS, run_solver  # noqa: E402

from corr_solver.cccp_mle_oracle import cccp_mle_oracle  # noqa: E402
from corr_solver.corr_oracle import (  # noqa: E402
    construct_distance_matrix,
    construct_poly_matrix,
    create_2d_sites,
)
from corr_solver.math_utils import mle_obj, omega_of  # noqa: E402

DEFAULT_OUT = Path(
    os.environ.get(
        "CORR_SOLVER_FIGS_DIR",
        "D:/github/luk036.github.io/cvx/lsq-vs-mle-remark.files",
    )
)

C = {
    "red": "#bf616a",
    "green": "#a3be8c",
    "blue": "#5e81ac",
    "yellow": "#ebcb8b",
    "purple": "#b48ead",
    "orange": "#d08770",
    "dark": "#2e3440",
    "grey": "#4c566a",
    "lred": "#d08a91",
    "lgreen": "#bcd3ab",
}
plt.rcParams.update(
    {"font.size": 11, "axes.edgecolor": C["grey"], "svg.fonttype": "none"}
)


def fig_corr(site, problems, rows, out: Path):
    D = construct_distance_matrix(site)
    iu = np.triu_indices_from(D)
    ds = D[iu]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (pname, Y) in zip(axes, problems.items()):
        lx, ly = PROBLEM_SPECS[pname]
        grid = np.linspace(0.0, float(ds.max()), 200)
        ax.scatter(ds, Y[iu], s=7, c=C["grey"], alpha=0.35, label="empirical")
        ax.scatter(
            ds,
            true_covariance(site, lx, ly)[iu],
            s=11,
            c="k",
            marker="x",
            alpha=0.45,
            label="true (generating)",
        )
        ax.plot(
            grid, true_kernel(grid, lx), "--", color="k", lw=1.5, label="true kernel x"
        )
        if ly != lx:
            ax.plot(
                grid,
                true_kernel(grid, ly),
                ":",
                color="k",
                lw=1.5,
                label="true kernel y",
            )
        for solver, col in [("LSQ-opt", C["blue"]), ("MLE", C["red"])]:
            r = next(r for r in rows if r["solver"] == solver and r["problem"] == pname)
            ax.plot(grid, r["fitted"](grid), lw=2.4, color=col, label=solver)
        ax.set_title(pname, color=C["dark"])
        ax.set_xlabel("Distance h")
        ax.set_ylabel("Covariance")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig_corr.svg", format="svg")
    plt.close(fig)


def fig_metrics(rows, out: Path):
    solvers = list(dict.fromkeys(r["solver"] for r in rows))
    problems = list(dict.fromkeys(r["problem"] for r in rows))
    palette = [C["red"], C["blue"], C["green"], C["yellow"], C["purple"]]
    x = np.arange(len(problems), dtype=float)
    width = 0.8 / len(solvers)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, (key, title, logy) in zip(
        axes,
        [
            ("rel_err", "Relative fit error", True),
            ("iters", "Iterations", False),
            ("time", "Runtime (s)", False),
        ],
    ):
        for k, solver in enumerate(solvers):
            vals = [
                next(
                    r[key] for r in rows if r["solver"] == solver and r["problem"] == p
                )
                for p in problems
            ]
            ax.bar(
                x + k * width,
                vals,
                width,
                label=solver,
                color=palette[k % len(palette)],
            )
        ax.set_xticks(x + width * (len(solvers) - 1) / 2)
        ax.set_xticklabels(problems, rotation=12, ha="right", fontsize=9)
        ax.set_title(title, color=C["dark"])
        if logy:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_metrics.svg", format="svg")
    plt.close(fig)


def fig_nonconvex(Y, out: Path):
    n = Y.shape[0]
    t = np.linspace(0.4, 4.0, 500)
    f2 = n * (2.0 - t) / t**3
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(t, f2, color=C["blue"], lw=3, zorder=3)
    ax.axhline(0.0, color=C["dark"], lw=1)
    ax.axvline(2.0, color=C["red"], ls="--", lw=2.2, zorder=2)
    ax.fill_between(
        t, f2, 0, where=(f2 > 0), color=C["green"], alpha=0.35, label="convex:  Ω ⪯ 2Y"
    )
    ax.fill_between(
        t, f2, 0, where=(f2 < 0), color=C["red"], alpha=0.30, label="concave:  Ω ⋡ 2Y"
    )
    ax.annotate(
        "t = 2  ⟺  Ω = 2Y",
        xy=(2.0, 0.35),
        xytext=(2.15, 0.75),
        color=C["red"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["red"]),
    )
    ax.set_xlabel("t   (Ω = t · Y)")
    ax.set_ylabel("f''(t) = n(2 − t) / t³")
    ax.set_title("The MLE objective is convex only on Ω ⪯ 2Y", color=C["dark"])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig_nonconvex.svg", format="svg")
    plt.close(fig)


def cccp_history(Y, site, m, x0, n_outer=30):
    Sig = construct_poly_matrix(site, m)
    x = np.array(x0, dtype=float)
    hist = [mle_obj(x, Sig, Y)]
    for _ in range(n_outer):
        M = np.linalg.inv(omega_of(x, Sig))
        oracle = cccp_mle_oracle(Sig, Y, M)
        x_new, _, _ = cutting_plane_optim(oracle, Ell(100.0, x), float("inf"))
        if x_new is None:
            break
        hist.append(mle_obj(x_new, Sig, Y))
        if abs(hist[-1] - hist[-2]) < 1e-8:
            break
        x = x_new
    return x, hist


def fig_cccp(Y, site, m, x_lsq, x_mle, out: Path):
    Sig = construct_poly_matrix(site, m)
    D = construct_distance_matrix(site)
    true = 4.0 * np.exp(-0.12 * D**2)

    def relerr(x):
        Om = sum(c * F for c, F in zip(x, Sig))
        return float(np.linalg.norm(true - Om, "fro") / np.linalg.norm(true, "fro"))

    x_cccp, hist = cccp_history(Y, site, m, x_lsq)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.3))

    ax1.plot(range(len(hist)), hist, "o-", color=C["blue"], lw=2.4, ms=5)
    ax1.axhline(
        cccp_obj_final := hist[-1],
        color=C["green"],
        ls="--",
        lw=1.5,
        label=f"free MLE = {cccp_obj_final:.3f}",
    )
    ax1.axhline(
        mle_obj(x_mle, Sig, Y),
        color=C["red"],
        ls=":",
        lw=1.8,
        label=f"MLE(2Y) = {mle_obj(x_mle, Sig, Y):.2f}",
    )
    ax1.set_xlabel("CCP outer iteration")
    ax1.set_ylabel("f(Ω) = log det Ω + Tr(Ω⁻¹Y)")
    ax1.set_title("CCP decreases f monotonically", color=C["dark"])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    methods = ["LSQ", "MLE(2Y)", "CCP"]
    xs = [x_lsq, x_mle, x_cccp]
    objs = [mle_obj(x, Sig, Y) for x in xs]
    rels = [relerr(x) for x in xs]
    xpos = np.arange(3)
    w = 0.38
    ax2.bar(xpos - w / 2, objs, w, color=C["purple"], label="MLE objective f")
    ax2b = ax2.twinx()
    ax2b.bar(xpos + w / 2, rels, w, color=C["orange"], label="rel. err vs true")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel("MLE objective f", color=C["purple"])
    ax2b.set_ylabel("relative error vs true kernel", color=C["orange"])
    ax2.set_title("CCP removes the 2Y-induced bias", color=C["dark"])
    ax2.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig_cccp.svg", format="svg")
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="directory to write the SVG figures into",
    )
    return parser.parse_args(argv)


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    site = create_2d_sites(5, 4)
    m = 4
    problems = build_problems(site)
    rows = []
    for pname, Y in problems.items():
        for sname in POLY_SOLVERS:
            r = run_solver(Y, site, m, sname, SOLVERS[sname])
            r["problem"] = pname
            rows.append(r)

    iso_Y = problems["iso (1,1)"]
    x_lsq = next(
        r["fitted"]
        for r in rows
        if r["solver"] == "LSQ-opt" and r["problem"] == "iso (1,1)"
    )
    x_mle = next(
        r["fitted"]
        for r in rows
        if r["solver"] == "MLE" and r["problem"] == "iso (1,1)"
    )

    fig_corr(site, problems, rows, out)
    fig_metrics(rows, out)
    fig_nonconvex(iso_Y, out)
    fig_cccp(iso_Y, site, m, np.asarray(x_lsq.c)[::-1], np.asarray(x_mle.c)[::-1], out)
    print("wrote:", *sorted(p.name for p in out.glob("fig_*.svg")))


if __name__ == "__main__":
    main(parse_args().out)
