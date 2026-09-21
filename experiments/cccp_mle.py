"""CCP (convex-concave procedure) solver for the family-constrained MLE.

The MLE objective f(Omega) = log det Omega + Tr(Omega^-1 Y) is a difference of
convex functions (h = Tr(Omega^-1 Y) convex, g = -log det Omega convex). CCP
majorizes f by linearizing g at the current iterate, yielding the convex
surrogate  Tr(Omega^-1 Y) + Tr(M_k Omega) + const,  solved each round with the
same cutting-plane machinery. This removes the hard 2Y constraint and the
non-convexity it was papering over.

The solver itself lives in :mod:`corr_solver.cccp_mle_oracle`; this script keeps
a polynomial-basis wrapper and a small demonstration.
"""

import numpy as np
from ellalgo.cutting_plane import cutting_plane_optim
from ellalgo.ell import Ell

from corr_solver.cccp_mle_oracle import cccp_mle as _cccp_mle
from corr_solver.cccp_mle_oracle import cccp_mle_oracle, mle_obj, omega_of
from corr_solver.corr_oracle import (
    construct_poly_matrix,
    corr_poly,
    create_2d_isotropic,
    create_2d_sites,
)
from corr_solver.lsq_corr_oracle import lsq_oracle
from corr_solver.mle_corr_oracle import mle_oracle

__all__ = [
    "cccp_mle",
    "cccp_mle_oracle",
    "lsq_core2",
    "make_expo_Y",
    "mle_core",
    "mle_obj",
    "omega_of",
]


def lsq_core2(Y, n, omega):
    normY = np.linalg.norm(Y, "fro")
    normY2 = 32 * normY * normY
    val = 256 * np.ones(n + 1)
    val[-1] = normY2 * normY2
    x = np.zeros(n + 1)
    x[0] = 1.0
    x[-1] = normY2 / 2
    xb, _, it = cutting_plane_optim(omega, Ell(val, x), float("inf"))
    return xb[:-1], it, True


def mle_core(_, n, omega):
    x = np.zeros(n)
    x[0] = 1.0
    xb, _, it = cutting_plane_optim(omega, Ell(50.0, x), float("inf"))
    return xb, it, xb is not None


def cccp_mle(Y, site, m, x0, n_outer=40, tol=1e-8):
    """Run the polynomial-basis CCP on a fixed site layout."""
    Sigma = construct_poly_matrix(site, m)
    return _cccp_mle(Y, Sigma, x0, n_outer=n_outer, tol=tol)


def make_expo_Y(site, N=3000):
    rng = np.random.RandomState(5)
    D = np.sqrt(np.sum((site[:, None, :] - site[None, :, :]) ** 2, axis=-1))
    A = np.linalg.cholesky(np.exp(-0.3 * D))
    n = site.shape[0]
    Y = np.zeros((n, n))
    for _ in range(N):
        y = A @ (2.0 * rng.randn(n)) + 1e-5 * rng.randn(n)
        Y += np.outer(y, y)
    return Y / N


def main():
    site = create_2d_sites(5, 4)
    m = 4
    Sigma = construct_poly_matrix(site, m)
    D = np.sqrt(np.sum((site[:, None, :] - site[None, :, :]) ** 2, axis=-1))

    cases = [
        (
            "gauss (mismatched)",
            create_2d_isotropic(site, 3000),
            4.0 * np.exp(-0.12 * D**2),
        ),
        ("expo (matched)", make_expo_Y(site), 4.0 * np.exp(-0.3 * D)),
    ]
    for name, Y, true in cases:
        x_lsq = corr_poly(Y, site, m, lsq_oracle, lsq_core2)[0].c[::-1]
        x_mle = corr_poly(Y, site, m, mle_oracle, mle_core)[0].c[::-1]
        x_cccp, n_outer = cccp_mle(Y, site, m, x_lsq)

        print(f"\n=== {name} ===")
        print(
            f"{'method':10s} {'f (MLE obj)':>12s} {'relerr vs true':>15s} {'2Y margin':>10s}"
        )
        for tag, x in [("LSQ", x_lsq), ("MLE(2Y)", x_mle), ("CCP", x_cccp)]:
            Om = omega_of(x, Sigma)
            margin = np.linalg.eigvalsh(2 * Y - Om).min()
            rel = np.linalg.norm(true - Om, "fro") / np.linalg.norm(true, "fro")
            print(f"{tag:10s} {mle_obj(x, Sigma, Y):12.4f} {rel:15.4f} {margin:10.4f}")
        print(f"CCP outer iterations: {n_outer}")


if __name__ == "__main__":
    main()
