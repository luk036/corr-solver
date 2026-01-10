# import necessary packages
import numpy as np
from matplotlib import pyplot as plt

# set random seed to ensure reproducibility
np.random.seed(8)


# define a kernel function to return a squared exponential distance between two input locations
def kernel(a, b):
    # decomposing the squaring operation into three parts
    #  each input location may be multi-dimensional, thus summing over all dimensions
    sq_dist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-sq_dist)


# setting number of input locations which approximates a function when growing to infinity
n = 200
X_test = np.linspace(-5, 5, n).reshape(-1, 1)

# calculate the pairwise distance, resulting in a nxn matrix
K = kernel(X_test, X_test)
# adding a small number along diagonal elements to ensure cholesky decomposition works
L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
# calculating functional samples by multiplying the sd with standard normal samples
samples = np.dot(L, np.random.normal(size=(n, 5)))
plt.plot(X_test, samples)
plt.show()


# Args:
#     X1: array of m points (m x d).
#     X2: array of n points (n x d).
# Returns:
#     (m x n) matrix.
def ise_kernel(X1, X2, length=1.0, sigma_f=1.0):
    sq_dist = (
        np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    )
    return sigma_f**2 * np.exp(-0.5 / length**2 * sq_dist)


# mean and covariance of the prior
mu = np.zeros(X_test.shape)
K = ise_kernel(X_test, X_test)

# draw samples from the prior using multivariate_normal from numpy
# convert mu from shape (n,1) to (n,)
samples = np.random.multivariate_normal(mean=mu.ravel(), cov=K, size=5)


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    # 95% confidence interval
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label="Mean")
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls="--", label=f"Sample {i + 1}")
    # plot observations if available
    if X_train is not None:
        plt.plot(X_train, Y_train, "rx")
    plt.legend()


plot_gp(mu, K, X_test, samples=samples)
plt.show()
