"""Fine sample-size sweep (N = 1..200) for LSQ, MLE(2Y), and CCP."""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common import cccp_run, make_Y, true_covariance  # noqa: E402

from corr_solver.corr_oracle import construct_poly_matrix, create_2d_sites  # noqa: E402
from corr_solver.lsq_corr_oracle import lsq_oracle  # noqa: E402
from corr_solver.math_utils import mle_obj  # noqa: E402
from corr_solver.mle_corr_oracle import mle_oracle  # noqa: E402
from corr_solver.solvers import lsq_corr_core2, mle_corr_core  # noqa: E402

NS = list(range(1, 21)) + [25, 30, 35, 40, 50, 60, 80, 100, 120, 150, 200]
PROBLEMS = [("iso (1,1)", 1.0, 1.0), ("aniso (1,3)", 1.0, 3.0)]
COLORS = {"LSQ": "#5e81ac", "MLE": "#bf616a", "CCP": "#a3be8c"}


def f3(v):
    return (
        "  --  " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f} "
    )


def sweep(site, m, lx, ly):
    true = true_covariance(site, lx, ly)
    normT = np.linalg.norm(true, "fro")
    Sig = construct_poly_matrix(site, m)
    rows = []
    for N in NS:
        Y = make_Y(site, lx, ly, N)
        ev = np.linalg.eigvalsh(Y)
        ymin, ymax = float(ev.min()), float(ev.max())
        cond = float(ymax / ymin) if ymin > 0 else float("inf")

        def metrics(x, feasible):
            if x is None or not feasible:
                return dict(relT=np.nan, margin=np.nan, obj=np.nan)
            Om = sum(c * F for c, F in zip(x, Sig))
            return dict(
                relT=float(np.linalg.norm(true - Om, "fro") / normT),
                margin=float(np.linalg.eigvalsh(2 * Y - Om).min()),
                obj=float(mle_obj(x, Sig, Y)),
            )

        try:
            xl, itl, fl = lsq_corr_core2(Y, m, lsq_oracle(Sig, Y))
        except Exception:
            xl, itl, fl = None, 0, False
        try:
            xm, itm, fm = mle_corr_core(Y, m, mle_oracle(Sig, Y))
        except Exception:
            xm, itm, fm = None, 0, False
        try:
            xc, itc = cccp_run(Y, site, m, xl) if fl else (None, 0)
        except Exception:
            xc, itc = None, 0

        rec = {"N": N, "ymin": ymin, "cond": cond}
        for tag, x, it, f in [
            ("LSQ", xl, itl, fl),
            ("MLE", xm, itm, fm),
            ("CCP", xc, itc, xc is not None),
        ]:
            rec[tag] = metrics(x, f)
            rec[tag]["iters"] = it
            rec[tag]["feasible"] = x is not None and f
        rows.append(rec)
        cs = "inf" if cond == float("inf") else f"{cond:.1e}"
        print(
            f"N={N:4d} minEigY={ymin:+.2e} cond={cs:>8s} | "
            f"LSQ {'ok ' if fl else 'NO '} relT={f3(rec['LSQ']['relT'])} it={itl:5d} | "
            f"MLE {'ok ' if fm else 'NO '} relT={f3(rec['MLE']['relT'])} it={itm:5d} | "
            f"CCP {'ok ' if rec['CCP']['feasible'] else 'NO '} relT={f3(rec['CCP']['relT'])} it={itc:5d}"
        )
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            os.environ.get(
                "CORR_SOLVER_FIGS_DIR",
                "D:/github/luk036.github.io/cvx/lsq-vs-mle-remark.files",
            )
        ),
        help="directory to write the SVG figures into",
    )
    return parser.parse_args(argv)


def main(out: Path):
    site = create_2d_sites(5, 4)
    m = 4
    results = {}
    for name, lx, ly in PROBLEMS:
        print(f"\n=== {name} ===")
        results[name] = sweep(site, m, lx, ly)

    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.4))
    for row, (name, _, _) in enumerate(PROBLEMS):
        rows = results[name]
        Ns = [r["N"] for r in rows]
        for tag in ["LSQ", "MLE", "CCP"]:
            axes[row, 0].plot(
                Ns,
                [r[tag]["relT"] for r in rows],
                "o-",
                ms=3,
                color=COLORS[tag],
                label=tag,
            )
            axes[row, 1].plot(
                Ns,
                [r[tag]["margin"] for r in rows],
                "o-",
                ms=3,
                color=COLORS[tag],
                label=tag,
            )
            axes[row, 2].plot(
                Ns,
                [r[tag]["iters"] for r in rows],
                "o-",
                ms=3,
                color=COLORS[tag],
                label=tag,
            )
            axes[row, 3].plot(
                Ns,
                [r[tag]["obj"] for r in rows],
                "o-",
                ms=3,
                color=COLORS[tag],
                label=tag,
            )
        axes[row, 0].set_ylabel(f"{name}\nrel. err vs true")
        axes[row, 1].axhline(0.0, color="#2e3440", lw=1)
        axes[row, 1].set_ylabel("min eig(2Y − Ω)")
        axes[row, 2].set_ylabel("iterations")
        axes[row, 3].set_ylabel("MLE objective f")
        for ax in axes[row]:
            ax.axvline(20, color="#b48ead", ls="--", lw=1.5)
            ax.set_xlabel("samples N")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
    axes[0, 0].set_title("Accuracy vs N (dashed = N = n = 20)", color="#2e3440")
    axes[0, 1].set_title("2Y margin", color="#2e3440")
    axes[0, 2].set_title(
        "Iterations (MLE wastes 2000 when infeasible)", color="#2e3440"
    )
    axes[0, 3].set_title("MLE objective", color="#2e3440")
    fig.tight_layout()
    fig.savefig(out / "fig_samples_fine.svg", format="svg")
    plt.close(fig)
    print("\nwrote fig_samples_fine.svg")


if __name__ == "__main__":
    main(parse_args().out)
