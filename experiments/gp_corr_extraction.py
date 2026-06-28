"""
Extract spatial correlation from 2D data via Gaussian Process Regression.
Test: generate isotropic 2D data from known kernel -> fit GPR -> verify + plot.
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


# ── step 1: generate isotropic 2D correlated data ─────────────────────


def sample_2d_grid(nx: int = 10, ny: int = 8) -> np.ndarray:
    xs = np.linspace(0, 10.0, nx)
    ys = np.linspace(0, 8.0, ny)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def true_kernel(kernel_type: str, length: float, sigma: float, noise: float) -> Kernel:
    base = {
        "gaussian": RBF(length_scale=length, length_scale_bounds="fixed"),
        "matern32": Matern(length_scale=length, nu=1.5, length_scale_bounds="fixed"),
        "matern52": Matern(length_scale=length, nu=2.5, length_scale_bounds="fixed"),
        "exponential": Matern(length_scale=length, nu=0.5, length_scale_bounds="fixed"),
    }[kernel_type]
    return ConstantKernel(sigma**2, constant_value_bounds="fixed") * base + WhiteKernel(
        noise_level=noise**2, noise_level_bounds="fixed"
    )


def simulate_isotropic_2d(
    X: np.ndarray,
    *,
    kernel_type: str = "gaussian",
    length: float = 2.0,
    sigma: float = 2.0,
    noise: float = 0.01,
    n_samples: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Return (n_locations x n_samples) array of correlated draws."""
    if rng is None:
        rng = np.random.default_rng(42)
    K = true_kernel(kernel_type, length, sigma, noise)(X)
    L = cholesky(K + 1e-10 * np.eye(len(X)), lower=True)
    return L @ rng.standard_normal((len(X), n_samples))


# ── multi-realisation MLE via sample covariance ──────────────────────


def _matern_kernel(r, sigma, nu):
    """Stable Matern evaluation avoiding inf*0 = NaN for large r."""
    if nu == 0.5:  # exponential
        return sigma**2 * np.exp(-r)
    r3 = np.sqrt(3) * r if nu == 1.5 else np.sqrt(5) * r
    if nu == 1.5:
        # (1 + r3) * exp(-r3), stable for all r3 >= 0
        logv = np.log1p(r3) - r3
    else:
        r5 = r3
        logv = np.log1p(r5 + r5**2 / 3) - r5
    # clip: when r3 is very large, result is effectively 0
    result = np.where(r < 50, sigma**2 * np.exp(logv), 0.0)
    return result


def build_signal_cov(X, kernel_type, length, sigma):
    d2 = np.sum((X[:, None, :] - X[None, :, :])**2, axis=-1)
    d = np.sqrt(d2 + 1e-12)
    if kernel_type == "gaussian":
        return sigma**2 * np.exp(-0.5 * d2 / length**2)
    r = d / length
    if kernel_type == "exponential":
        return _matern_kernel(r, sigma, 0.5)
    if kernel_type == "matern32":
        return _matern_kernel(r, sigma, 1.5)
    if kernel_type == "matern52":
        return _matern_kernel(r, sigma, 2.5)
    raise ValueError(f"unknown kernel: {kernel_type}")


def corr_fcn(h, kernel_type, length, sigma, noise):
    h = np.asarray(h, dtype=float)
    r = h / length
    if kernel_type == "gaussian":
        c = sigma**2 * np.exp(-0.5 * h**2 / length**2)
    elif kernel_type == "exponential":
        c = _matern_kernel(r, sigma, 0.5)
    elif kernel_type == "matern32":
        c = _matern_kernel(r, sigma, 1.5)
    elif kernel_type == "matern52":
        c = _matern_kernel(r, sigma, 2.5)
    else:
        raise ValueError(f"unknown kernel: {kernel_type}")
    c[h == 0] += noise**2
    return c


def nlml_multi(params, X, Y, n_samples, kernel_type):
    """Negative log marginal likelihood (per-sample average).

    All params on log-scale: [log(length), log(sigma), log(noise)].
    Returns -log L / M.
    """
    log_len, log_sig, log_noi = np.clip(params, [-3, -5, -12], [4, 5, 2])
    K = build_signal_cov(X, kernel_type, np.exp(log_len), np.exp(log_sig))
    noise2 = np.exp(2 * log_noi)
    K[np.diag_indices_from(K)] += noise2 + 1e-10
    if np.any(np.isnan(K)) or np.any(np.isinf(K)):
        return 1e12
    try:
        L = cholesky(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e12
    A = solve_triangular(L.T, solve_triangular(L, Y, lower=True))
    n = len(X)
    return 0.5 * (np.trace(A) + 2 * np.sum(np.log(np.diag(L))) + n * np.log(2 * np.pi))


def extract_multi_mle(X, Y, n_samples, kernel_type, true_length=None):
    """Multi-start L-BFGS-B on per-sample nlml."""
    bounds = [(-1.5, 3.5), (-3, 3), (-12, 2)]  # log-scale

    starts = [[np.log(2.0), np.log(1.0), np.log(0.1)]]
    if true_length is not None:
        starts.append([np.log(true_length), np.log(2.0), np.log(0.01)])
    for _ in range(10):
        starts.append(np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds]))

    best = {"nlml_per_sample": float("inf")}
    for x0 in starts:
        res = minimize(
            nlml_multi, x0, args=(X, Y, n_samples, kernel_type),
            method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 500},
        )
        if res.fun < best["nlml_per_sample"]:
            est = np.exp(res.x)
            best = {
                "length": est[0], "sigma": est[1], "noise": est[2],
                "nlml_per_sample": res.fun,
            }
    return best


# ── single-realisation GPR (sklearn) ─────────────────────────────────


def est_kernel(kernel_type: str) -> Kernel:
    amp = ConstantKernel(1.0, (1e-3, 1e3))
    base = {
        "gaussian": RBF(1.0, (1e-2, 1e2)),
        "matern32": Matern(1.0, nu=1.5, length_scale_bounds=(1e-2, 1e2)),
        "matern52": Matern(1.0, nu=2.5, length_scale_bounds=(1e-2, 1e2)),
        "exponential": Matern(1.0, nu=0.5, length_scale_bounds=(1e-2, 1e2)),
    }[kernel_type]
    return amp * base + WhiteKernel(0.1, (1e-6, 1e2))


def extract_correlation_gpr(X, y, kernel_type, n_restarts=5):
    gp = GaussianProcessRegressor(
        kernel=est_kernel(kernel_type),
        n_restarts_optimizer=n_restarts,
        random_state=42,
    )
    gp.fit(X, y)
    p = gp.kernel_.get_params()
    length = p.get("k1__k2__length_scale", p.get("k1__length_scale", np.nan))
    sigma = np.sqrt(p.get("k1__k1__constant_value", p.get("k1__constant_value", np.nan)))
    noise = np.sqrt(p.get("k2__noise_level", np.nan))
    return {"length": length, "sigma": sigma, "noise": noise}


# ── plot: correlation vs distance ────────────────────────────────────


def plot_correlation_comparison(true, est_gpr, est_mle, kernel_type, ax):
    hs = np.linspace(0, 15, 200)
    c_true = corr_fcn(hs, kernel_type, true["length"], true["sigma"], true["noise"])
    c_gpr = corr_fcn(hs, kernel_type, est_gpr["length"], est_gpr["sigma"], est_gpr["noise"])
    c_mle = corr_fcn(hs, kernel_type, est_mle["length"], est_mle["sigma"], est_mle["noise"])

    ax.plot(hs, c_true, "k-", linewidth=2.5, label="True")
    ax.plot(hs, c_gpr, "b--", label=f"GPR (len={est_gpr['length']:.2f})")
    ax.plot(hs, c_mle, "r:", label=f"MLE (len={est_mle['length']:.2f})")
    ax.set_xlabel("Distance h")
    ax.set_ylabel("Covariance C(h)")
    ax.set_title(f"{kernel_type} kernel  (true len={true['length']})")
    ax.legend()
    ax.grid(True, alpha=0.3)


# ── verify + plot ────────────────────────────────────────────────────


def run_case(
    kernel_type: str = "gaussian",
    length_true: float = 2.0,
    sigma_true: float = 2.0,
    noise_true: float = 0.01,
    n_samples: int = 2000,
    save_plot: bool = True,
) -> dict:
    X = sample_2d_grid()
    Y_samples = simulate_isotropic_2d(
        X, kernel_type=kernel_type, length=length_true,
        sigma=sigma_true, noise=noise_true, n_samples=n_samples,
    )
    gpr = extract_correlation_gpr(X, Y_samples[:, 0], kernel_type)
    Y_cov = Y_samples @ Y_samples.T / n_samples
    mle = extract_multi_mle(X, Y_cov, n_samples, kernel_type, true_length=length_true)

    true = {"length": length_true, "sigma": sigma_true, "noise": noise_true}

    if save_plot and HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: correlation function C(h)
        plot_correlation_comparison(true, gpr, mle, kernel_type, axes[0])

        # Right: empirical vs fitted covariance matrix (1st row)
        n = len(X)
        h_idx = np.argsort(np.sqrt(np.sum((X - X[0])**2, axis=1)))
        axes[1].plot(Y_cov[0, h_idx], 'k.-', label="Empirical", alpha=0.6)
        axes[1].plot(
            build_signal_cov(X, kernel_type, mle["length"], mle["sigma"])[0, h_idx]
            + mle["noise"]**2 * (h_idx == 0),
            'r.-', label="Fitted MLE", alpha=0.7,
        )
        axes[1].set_xlabel("Site index (sorted by distance from site 0)")
        axes[1].set_ylabel("Covariance")
        axes[1].set_title(f"Row 0 of covariance matrix")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"corr_{kernel_type}_len{length_true}.png"
        plt.savefig(fname, dpi=120)
        print(f"     [plot saved: {fname}]")
        plt.close()

    return {"gpr": gpr, "mle": mle, "true": true}


def demo():
    suites = [
        ("gaussian", 2.0, 2.0, 0.01),
        ("matern32", 2.0, 2.0, 0.01),
        ("matern52", 2.0, 2.0, 0.01),
        ("exponential", 2.0, 2.0, 0.01),
        ("gaussian", 1.0, 1.0, 0.001),
        ("matern32", 3.0, 1.5, 0.05),
    ]

    for kt, ell, sg, ns in suites:
        print(f"\n  -- {kt}: true len={ell}  sig={sg}  noise={ns}")
        res = run_case(kt, ell, sg, ns)
        g, m = res["gpr"], res["mle"]
        print(f"     GPR: len={g['length']:.3f}  sig={g['sigma']:.3f}  noise={g['noise']:.5f}")
        print(f"     MLE: len={m['length']:.3f}  sig={m['sigma']:.3f}  noise={m['noise']:.5f}")


if __name__ == "__main__":
    demo()
