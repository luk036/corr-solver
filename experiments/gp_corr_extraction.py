"""
Extract spatial correlation from 2D data via Gaussian Process Regression.
Supports isotropic (single length) and anisotropic (x-length, y-length) kernels.
Test: generate 2D data from known kernel -> fit GPR -> verify + plot.
"""

from typing import Optional

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Kernel,
    Matern,
    RBF,
    WhiteKernel,
)

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── helpers ────────────────────────────────────────────────────────────


def sample_2d_grid(nx: int = 10, ny: int = 8) -> np.ndarray:
    xs = np.linspace(0, 10.0, nx)
    ys = np.linspace(0, 8.0, ny)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _matern_kernel(r, sigma, nu):
    if nu == 0.5:
        return sigma**2 * np.exp(-r)
    r3 = np.sqrt(3) * r if nu == 1.5 else np.sqrt(5) * r
    logv = np.log1p(r3) - r3 if nu == 1.5 else np.log1p(r3 + r3**2 / 3) - r3
    return np.where(r < 50, sigma**2 * np.exp(logv), 0.0)


def _eval_kernel(r2, kernel_type, sigma):
    r = np.sqrt(r2 + 1e-12)
    if kernel_type == "gaussian":
        return sigma**2 * np.exp(-0.5 * r2)
    if kernel_type == "exponential":
        return _matern_kernel(r, sigma, 0.5)
    if kernel_type == "matern32":
        return _matern_kernel(r, sigma, 1.5)
    if kernel_type == "matern52":
        return _matern_kernel(r, sigma, 2.5)
    raise ValueError(f"unknown kernel: {kernel_type}")


def _extract_lengths(p, key="k1__k2__length_scale", fallback="k1__length_scale"):
    v = p.get(key, p.get(fallback, np.nan))
    if hasattr(v, "__iter__"):
        return float(v[0]), float(v[1])
    return float(v), float(v)


# ── step 1: generate 2D correlated data ─────────────────────────────


def make_true_kernel(kernel_type, length1, length2, sigma, noise):
    """Build a fixed sklearn kernel (isotropic or anisotropic)."""
    ls = length1 if length1 == length2 else [length1, length2]
    base = {
        "gaussian": RBF(length_scale=ls, length_scale_bounds="fixed"),
        "matern32": Matern(length_scale=ls, nu=1.5, length_scale_bounds="fixed"),
        "matern52": Matern(length_scale=ls, nu=2.5, length_scale_bounds="fixed"),
        "exponential": Matern(length_scale=ls, nu=0.5, length_scale_bounds="fixed"),
    }[kernel_type]
    return ConstantKernel(sigma**2, constant_value_bounds="fixed") * base \
           + WhiteKernel(noise**2, noise_level_bounds="fixed")


def _simulate(X, kernel_type, length1, length2, sigma, noise, n_samples, rng):
    if rng is None:
        rng = np.random.default_rng(42)
    K = make_true_kernel(kernel_type, length1, length2, sigma, noise)(X)
    L = cholesky(K + 1e-10 * np.eye(len(X)), lower=True)
    return L @ rng.standard_normal((len(X), n_samples))


# ── build signal covariance (for multi-realisation MLE) ──────────────


def aniso_r2(X, length1, length2):
    """Anisotropic squared distance: (dx/l1)^2 + (dy/l2)^2."""
    dx = X[:, None, 0] - X[None, :, 0]
    dy = X[:, None, 1] - X[None, :, 1]
    return (dx / length1)**2 + (dy / length2)**2


def build_signal_cov_aniso(X, kernel_type, length1, length2, sigma):
    return _eval_kernel(aniso_r2(X, length1, length2), kernel_type, sigma)


def build_signal_cov(X, kernel_type, length, sigma):
    """Isotropic: calls anisotropic with length1=length2=length."""
    return build_signal_cov_aniso(X, kernel_type, length, length, sigma)


def nlml_multi_aniso(params, X, Y, n_samples, kernel_type):
    """4 params on log-scale: [log(l1), log(l2), log(sigma), log(noise)]."""
    log_l1, log_l2, log_sig, log_noi = np.clip(params, [-3, -3, -5, -12], [4, 4, 5, 2])
    K = build_signal_cov_aniso(X, kernel_type, np.exp(log_l1), np.exp(log_l2), np.exp(log_sig))
    K[np.diag_indices_from(K)] += np.exp(2 * log_noi) + 1e-10
    if np.any(np.isnan(K)) or np.any(np.isinf(K)):
        return 1e12
    try:
        L = cholesky(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e12
    A = solve_triangular(L.T, solve_triangular(L, Y, lower=True))
    n = len(X)
    return 0.5 * (np.trace(A) + 2 * np.sum(np.log(np.diag(L))) + n * np.log(2 * np.pi))


def extract_multi_mle_aniso(X, Y, n_samples, kernel_type, true_l1=None, true_l2=None):
    """Multi-start L-BFGS-B (4 params: l1, l2, sigma, noise)."""
    bounds = [(-1.5, 3.5), (-1.5, 3.5), (-3, 3), (-12, 2)]
    starts = [[np.log(2.0), np.log(2.0), np.log(1.0), np.log(0.1)]]
    if true_l1 is not None and true_l2 is not None:
        starts.append([np.log(true_l1), np.log(true_l2), np.log(2.0), np.log(0.01)])
    for _ in range(12):
        starts.append(np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds]))

    best = {"nlml_per_sample": float("inf")}
    for x0 in starts:
        res = minimize(
            nlml_multi_aniso, x0, args=(X, Y, n_samples, kernel_type),
            method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 500},
        )
        if res.fun < best["nlml_per_sample"]:
            est = np.exp(res.x)
            best = {"length1": est[0], "length2": est[1],
                    "sigma": est[2], "noise": est[3],
                    "nlml_per_sample": res.fun}
    return best


# ── single-realisation GPR (sklearn) ─────────────────────────────────


def est_kernel_aniso(kernel_type: str) -> Kernel:
    """Anisotropic trainable kernel: separate length per dimension."""
    amp = ConstantKernel(1.0, (1e-3, 1e3))
    ls_bounds = (1e-2, 1e2)
    base = {
        "gaussian": RBF([1.0, 1.0], length_scale_bounds=ls_bounds),
        "matern32": Matern([1.0, 1.0], nu=1.5, length_scale_bounds=ls_bounds),
        "matern52": Matern([1.0, 1.0], nu=2.5, length_scale_bounds=ls_bounds),
        "exponential": Matern([1.0, 1.0], nu=0.5, length_scale_bounds=ls_bounds),
    }[kernel_type]
    return amp * base + WhiteKernel(0.1, (1e-6, 1e2))


def extract_correlation_gpr_aniso(X, y, kernel_type, n_restarts=5):
    gp = GaussianProcessRegressor(
        kernel=est_kernel_aniso(kernel_type),
        n_restarts_optimizer=n_restarts,
        random_state=42,
    )
    gp.fit(X, y)
    p = gp.kernel_.get_params()
    l1, l2 = _extract_lengths(p)
    sigma = np.sqrt(p.get("k1__k1__constant_value", p.get("k1__constant_value", np.nan)))
    noise = np.sqrt(p.get("k2__noise_level", np.nan))
    return {"length1": l1, "length2": l2, "sigma": sigma, "noise": noise}


# ── isotropic wrappers (backward compat) ─────────────────────────────


def simulate_isotropic_2d(X, *, kernel_type="gaussian", length=2.0,
                          sigma=2.0, noise=0.01, n_samples=2000, rng=None):
    return _simulate(X, kernel_type, length, length, sigma, noise, n_samples, rng)


def simulate_anisotropic_2d(X, *, kernel_type="gaussian", length1=2.0, length2=5.0,
                            sigma=2.0, noise=0.01, n_samples=2000, rng=None):
    return _simulate(X, kernel_type, length1, length2, sigma, noise, n_samples, rng)


def corr_fcn_aniso(dx, dy, kernel_type, length1, length2, sigma, noise):
    r2 = (dx / length1)**2 + (dy / length2)**2
    c = _eval_kernel(r2, kernel_type, sigma)
    c[(dx == 0) & (dy == 0)] += noise**2
    return c


# ── isotropic wrappers for backward compat ──────────────────────────


def extract_multi_mle(X, Y, n_samples, kernel_type, true_length=None):
    r = extract_multi_mle_aniso(X, Y, n_samples, kernel_type,
                                true_l1=true_length, true_l2=true_length)
    r["length"] = r["length1"]
    return r


def extract_correlation_gpr(X, y, kernel_type, n_restarts=5):
    r = extract_correlation_gpr_aniso(X, y, kernel_type, n_restarts)
    r["length"] = r["length1"]
    return r


def build_signal_cov_iso(X, kernel_type, length, sigma):
    return build_signal_cov_aniso(X, kernel_type, length, length, sigma)


def corr_fcn(h, kernel_type, length, sigma, noise):
    return corr_fcn_aniso(h, np.zeros_like(h) if hasattr(h, "__len__") else 0.0,
                          kernel_type, length, length, sigma, noise)


# ── plotting ─────────────────────────────────────────────────────────


def plot_correlation_comparison(true, est_gpr, est_mle, kernel_type, ax):
    hs = np.linspace(0, 15, 200)
    c_true = corr_fcn(hs, kernel_type, true["length1"], true["sigma"], true["noise"])
    c_gpr = corr_fcn(hs, kernel_type, est_gpr["length1"], est_gpr["sigma"], est_gpr["noise"])
    c_mle = corr_fcn(hs, kernel_type, est_mle["length1"], est_mle["sigma"], est_mle["noise"])
    ax.plot(hs, c_true, "k-", lw=2.5, label="True")
    ax.plot(hs, c_gpr, "b--", label=f"GPR (l={est_gpr['length1']:.2f})")
    ax.plot(hs, c_mle, "r:", label=f"MLE (l={est_mle['length1']:.2f})")
    ax.set_xlabel("Distance h"); ax.set_ylabel("Covariance C(h)")
    ax.set_title(f"{kernel_type}  true l1={true['length1']}")
    ax.legend(); ax.grid(True, alpha=0.3)


def run_case(kernel_type="gaussian", length_true=2.0, sigma_true=2.0,
             noise_true=0.01, n_samples=2000, save_plot=True):
    X = sample_2d_grid()
    Y_samples = simulate_isotropic_2d(
        X, kernel_type=kernel_type, length=length_true,
        sigma=sigma_true, noise=noise_true, n_samples=n_samples,
    )
    gpr = extract_correlation_gpr(X, Y_samples[:, 0], kernel_type)
    Y_cov = Y_samples @ Y_samples.T / n_samples
    mle = extract_multi_mle(X, Y_cov, n_samples, kernel_type, true_length=length_true)
    true = {"length1": length_true, "length2": length_true,
            "sigma": sigma_true, "noise": noise_true}

    if save_plot and HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_correlation_comparison(true, gpr, mle, kernel_type, axes[0])
        n = len(X)
        h_idx = np.argsort(np.sqrt(np.sum((X - X[0])**2, axis=1)))
        axes[1].plot(Y_cov[0, h_idx], 'k.-', label="Empirical", alpha=0.6)
        mleK = build_signal_cov_aniso(X, kernel_type, mle["length1"], mle["length2"], mle["sigma"])
        axes[1].plot(mleK[0, h_idx] + (h_idx == 0) * mle["noise"]**2,
                     'r.-', label="Fitted MLE", alpha=0.7)
        axes[1].set_xlabel("Site index"); axes[1].set_ylabel("Covariance")
        axes[1].set_title("Row 0 of covariance matrix")
        axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"corr_{kernel_type}_len{length_true}.png", dpi=120)
        plt.close()
    return {"gpr": gpr, "mle": mle, "true": true}


# ── anisotropic demo ─────────────────────────────────────────────────


def demo_aniso():
    X = sample_2d_grid()
    print("=" * 60)
    print("  Anisotropic Correlation Extraction")
    print("  True: l1=/l2= differ (x vs y direction)")
    print("=" * 60)

    cases = [
        ("gaussian", 1.0, 4.0, 2.0, 0.01),
        ("gaussian", 3.0, 1.0, 2.0, 0.01),
        ("gaussian", 1.5, 6.0, 1.5, 0.01),
        ("matern32", 1.0, 4.0, 2.0, 0.01),
        ("matern52", 4.0, 1.0, 2.0, 0.01),
        ("exponential", 2.0, 6.0, 2.0, 0.01),
    ]

    for kt, ell1, ell2, sg, ns in cases:
        print(f"\n  -- {kt}: l1={ell1}  l2={ell2}  sig={sg}  noise={ns}")
        Y = simulate_anisotropic_2d(
            X, kernel_type=kt, length1=ell1, length2=ell2,
            sigma=sg, noise=ns, n_samples=200,
        )
        gpr = extract_correlation_gpr_aniso(X, Y[:, 0], kt)
        Yc = Y @ Y.T / 200
        mle = extract_multi_mle_aniso(X, Yc, 200, kt, true_l1=ell1, true_l2=ell2)
        print(f"     GPR: l1={gpr['length1']:.3f}  l2={gpr['length2']:.3f}  "
              f"sig={gpr['sigma']:.3f}  noise={gpr['noise']:.5f}")
        print(f"     MLE: l1={mle['length1']:.3f}  l2={mle['length2']:.3f}  "
              f"sig={mle['sigma']:.3f}  noise={mle['noise']:.5f}")


def demo():
    print("=" * 60)
    print("  Isotropic demo (backward compat)")
    print("=" * 60)
    for kt, ell, sg, ns in [
        ("gaussian", 2.0, 2.0, 0.01),
        ("matern32", 2.0, 2.0, 0.01),
        ("matern52", 2.0, 2.0, 0.01),
        ("exponential", 2.0, 2.0, 0.01),
    ]:
        print(f"\n  -- {kt}: len={ell}  sig={sg}  noise={ns}")
        res = run_case(kt, ell, sg, ns)
        g, m = res["gpr"], res["mle"]
        print(f"     GPR: len={g['length']:.3f}  sig={g['sigma']:.3f}  noise={g['noise']:.5f}")
        print(f"     MLE: len={m['length']:.3f}  sig={m['sigma']:.3f}  noise={m['noise']:.5f}")


if __name__ == "__main__":
    demo()
    print("\n")
    demo_aniso()
