"""Shared helpers for the correlation-solver experiments.

The reusable cutting-plane cores live in :mod:`corr_solver.solvers`; this module
only holds the experiment-side primitives: the covariance generators, the true
kernels and the CCP outer loop.
"""

import numpy as np
from ellalgo.cutting_plane import cutting_plane_optim
from ellalgo.ell import Ell

from corr_solver.cccp_mle_oracle import cccp_mle_oracle
from corr_solver.corr_oracle import construct_poly_matrix, create_2d_isotropic
from corr_solver.kernels import gaussian
from corr_solver.math_utils import mle_obj, omega_of

SDKERN = 0.12
VAR = 2.0
TAU = 0.00001


PROBLEM_SPECS = {
    "iso (1,1)": (1.0, 1.0),
    "aniso (1,3)": (1.0, 3.0),
    "aniso (3,1)": (3.0, 1.0),
}


def create_2d_anisotropic(
    site, length_x, length_y, N=3000, rng=None
):
    """Biased sample covariance from the anisotropic Gaussian kernel."""
    n = site.shape[0]
    if rng is None:
        rng = np.random.RandomState(5)

    dx = site[:, None, 0] - site[None, :, 0]
    dy = site[:, None, 1] - site[None, :, 1]
    dist_sq = (dx / length_x) ** 2 + (dy / length_y) ** 2
    Sigma = gaussian(dist_sq, SDKERN)

    A = np.linalg.cholesky(Sigma)
    Y = np.zeros((n, n))
    outer_buf = np.empty((n, n))
    for _ in range(N):
        y = A @ (VAR * rng.randn(n)) + TAU * rng.randn(n)
        np.outer(y, y, out=outer_buf)
        Y += outer_buf
    return Y / N


def true_covariance(site, length_x, length_y):
    """Noise-free generating covariance of the anisotropic Gaussian kernel."""
    dx = site[:, None, 0] - site[None, :, 0]
    dy = site[:, None, 1] - site[None, :, 1]
    dist_sq = (dx / length_x) ** 2 + (dy / length_y) ** 2
    C = (VAR**2) * gaussian(dist_sq, SDKERN)
    np.fill_diagonal(C, C.diagonal() + TAU**2)
    return C


def true_kernel(h, length):
    """One-dimensional radial profile of the generating kernel."""
    return (VAR**2) * gaussian((h / length) ** 2, SDKERN)


def mle_objective(omega, Y):
    """Return ``log det omega + Tr(omega^-1 Y)``, or ``nan`` if not positive definite."""
    sign, logdet = np.linalg.slogdet(omega)
    if sign <= 0:
        return float("nan")
    return float(logdet + np.trace(np.linalg.solve(omega, Y)))


def make_Y(site, lx, ly, N):
    """Sample covariance, isotropic when ``lx == ly``."""
    if lx == ly:
        return create_2d_isotropic(site, N)
    return create_2d_anisotropic(site, lx, ly, N)


def build_problems(site, N=3000):
    """Build one biased sample covariance per entry of :data:`PROBLEM_SPECS`."""
    out = {}
    for name, (length_x, length_y) in PROBLEM_SPECS.items():
        out[name] = make_Y(site, length_x, length_y, N)
    return out


def cccp_run(Y, site, m, x0, n_outer=15):
    """Run the polynomial-basis CCP, returning the iterate and total inner iterations."""
    Sig = construct_poly_matrix(site, m)
    x = np.array(x0, dtype=float)
    f_old = np.inf
    total = 0
    for _ in range(n_outer):
        M = np.linalg.inv(omega_of(x, Sig))
        oracle = cccp_mle_oracle(Sig, Y, M)
        x_new, _, it = cutting_plane_optim(oracle, Ell(100.0, x), float("inf"))
        total += it
        if x_new is None:
            return None, total
        f_new = mle_obj(x_new, Sig, Y)
        if abs(f_old - f_new) < 1e-8:
            return x_new, total
        f_old = f_new
        x = x_new
    return x, total
