"""Sample-size study: how N affects LSQ, MLE(2Y), and CCP (isotropic + anisotropic)."""

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
from corr_solver.math_utils import mle_obj, omega_of  # noqa: E402
from corr_solver.mle_corr_oracle import mle_oracle  # noqa: E402
from corr_solver.solvers import lsq_corr_core2, mle_corr_core  # noqa: E402

NS = [5, 10, 15, 20, 30, 50, 100, 200, 500, 1000, 2000]
PROBLEMS = [("iso (1,1)", 1.0, 1.0), ("aniso (1,3)", 1.0, 3.0)]
COLORS = {"LSQ": "#5e81ac", "MLE": "#bf616a", "CCP": "#a3be8c"}


def sweep(site, m, lx, ly):
    true = true_covariance(site, lx, ly)
    normT = np.linalg.norm(true, "fro")
    Sig = construct_poly_matrix(site, m)
    rows = []
    for N in NS:
        Y = make_Y(site, lx, ly, N)

        def metrics(x, feasible):
            if x is None or not feasible:
                return dict(relT=np.nan, margin=np.nan, obj=np.nan)
            Om = omega_of(x, Sig)
            return dict(
                relT=float(np.linalg.norm(true - Om, "fro") / normT),
                margin=float(np.linalg.eigvalsh(2 * Y - Om).min()),
                obj=float(mle_obj(x, Sig, Y)),
            )

        try:
            xl, _, fl = lsq_corr_core2(Y, m, lsq_oracle(Sig, Y))
        except Exception:
            xl, fl = None, False
        try:
            xm, _, fm = mle_corr_core(Y, m, mle_oracle(Sig, Y))
        except Exception:
            xm, fm = None, False
        try:
            xc = cccp_run(Y, site, m, xl)[0] if fl else None
        except Exception:
            xc = None

        rec = {"N": N, "ymin": float(np.linalg.eigvalsh(Y).min())}
        for tag, x, f in [
            ("LSQ", xl, fl),
            ("MLE", xm, fm),
            ("CCP", xc, xc is not None),
        ]:
            rec[tag] = metrics(x, f)
        rows.append(rec)
        print(
            f"N={N:5d} minEigY={rec['ymin']:+.2e} | "
            f"LSQ {rec['LSQ']['relT']:.3f} (m {rec['LSQ']['margin']:+.2f}) | "
            f"MLE {rec['MLE']['relT']:.3f} (m {rec['MLE']['margin']:+.2f}, f {rec['MLE']['obj']:.1f}) | "
            f"CCP {rec['CCP']['relT']:.3f} (f {rec['CCP']['obj']:.1f})"
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
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.2))
    for row, (name, _, _) in enumerate(PROBLEMS):
        rows = results[name]
        Ns = [r["N"] for r in rows]
        for tag in ["LSQ", "MLE", "CCP"]:
            axes[row, 0].plot(
                Ns, [r[tag]["relT"] for r in rows], "o-", color=COLORS[tag], label=tag
            )
            axes[row, 1].plot(
                Ns, [r[tag]["margin"] for r in rows], "o-", color=COLORS[tag], label=tag
            )
            axes[row, 2].plot(
                Ns, [r[tag]["obj"] for r in rows], "o-", color=COLORS[tag], label=tag
            )
        axes[row, 0].set_ylabel(f"{name}\nrel. err vs true")
        axes[row, 1].axhline(0.0, color="#2e3440", lw=1)
        axes[row, 1].set_ylabel("min eig(2Y − Ω)")
        axes[row, 2].set_ylabel("MLE objective f")
        for ax in axes[row]:
            ax.set_xscale("log")
            ax.set_xlabel("samples N")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
    axes[0, 0].set_title("Accuracy vs N", color="#2e3440")
    axes[0, 1].set_title(
        "2Y margin: pinned at 0 ⇒ constraint always active", color="#2e3440"
    )
    axes[0, 2].set_title("MLE objective", color="#2e3440")
    fig.tight_layout()
    fig.savefig(out / "fig_samples.svg", format="svg")
    plt.close(fig)
    print("\nwrote fig_samples.svg")


if __name__ == "__main__":
    main(parse_args().out)
