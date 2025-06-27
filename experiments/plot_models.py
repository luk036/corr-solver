import numpy as np

from corr_solver.corr_oracle import create_2d_sites

Arr = np.ndarray


def plot_2d_isotropic(site: Arr) -> None:
    n = site.shape[0]
    sdkern = 0.12  # width of kernel
    var = 2.0  # standard derivation
    tau = 0.00001  # standard derivation of white noise
    np.random.seed(5)

    Sigma = np.zeros((n, n))
    for row in range(n):
        for col in range(row, n):
            d = np.array(site[col]) - np.array(site[row])
            Sigma[row, col] = np.exp(-sdkern * (d.dot(d)))
            Sigma[col, row] = Sigma[row, col]

    A = np.linalg.cholesky(Sigma)

    for _ in range(4):
        x = var * np.random.randn(n)
        y = A @ x + tau * np.random.randn(n)


site = create_2d_sites(5, 4)
plot_2d_isotropic(site, 3000)
