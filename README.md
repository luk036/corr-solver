# 🖇️ corr-solver

![Python application](https://github.com/luk036/corr-solver/workflows/Python%20application/badge.svg)
[![Python application w/ Coverage](https://github.com/luk036/corr-solver/actions/workflows/python-app.yml/badge.svg)](https://github.com/luk036/corr-solver/actions/workflows/python-app.yml)
[![Multi-Platforms](https://github.com/luk036/corr-solver/actions/workflows/multi-platforms.yml/badge.svg)](https://github.com/luk036/corr-solver/actions/workflows/multi-platforms.yml)
[![Documentation Status](https://readthedocs.org/projects/corr-solver/badge/?version=latest)](https://corr-solver.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/a2f75bd3cc1e4c34be4741bdd61168ba)](https://app.codacy.com/app/luk036/corr-solver?utm_source=github.com&utm_medium=referral&utm_content=luk036/corr-solver&utm_campaign=badger)
[![codecov](https://codecov.io/gh/luk036/corr-solver/graph/badge.svg?token=Zuh7Egf1Rk)](https://codecov.io/gh/luk036/corr-solver)

> Correlation Solver Python Code

This code defines a class called mle_oracle which is designed to solve a maximum likelihood estimation problem. The purpose of this code is to find the best parameters for a statistical model based on observed data, while satisfying certain constraints.

The mle_oracle class takes two inputs when initialized: Sigma and Y. Sigma represents a covariance matrix, which describes how different variables in a dataset are related to each other. Y is a biased sample covariance matrix, which is an estimate of the true covariance based on observed data.

The main output of this class is produced by the assess_optim method. This method takes two inputs: x (a set of coefficients) and t (the best optimal value found so far). It returns a tuple containing information about whether the current solution is feasible and optimal, along with some additional values used in the optimization process.

To achieve its purpose, the code uses a technique called linear matrix inequality (LMI) optimization. It creates two LMI oracles (lmi0 and lmi) which are used to check if the current solution satisfies certain constraints. The assess_optim method first checks if the solution is feasible using these oracles. If it's not feasible, it returns information about why it's not feasible.

If the solution is feasible, the method then calculates a value f1, which represents the objective function of the maximum likelihood estimation problem. This calculation involves matrix operations like inversion, multiplication, and calculating traces and determinants. The method also computes a gradient g, which indicates how the objective function changes with respect to small changes in the input x.

Finally, the method compares the calculated f1 with the input t to determine if a better solution has been found. If f1 is better than t, it returns this new value along with the gradient. Otherwise, it returns information that can be used to continue the optimization process.

The important logic flows in this code include the feasibility checks, the calculation of the objective function and its gradient, and the comparison of the current solution with the best known solution. The data transformations mainly involve matrix operations on the input covariance matrices.

Overall, this code provides a way to solve a complex statistical optimization problem by iteratively improving a solution while ensuring it satisfies certain constraints. It's a building block that would typically be used as part of a larger optimization algorithm.

## Dependencies

- [luk036/lds-gen](https://github.com/luk036/lds-gen)
- [luk036/ellalgo](https://github.com/luk036/ellalgo)
- numpy
- scipy

## ✨ Features

- Explore convexity

## 🛠️ Installation

- The core corr-solver depends on the `lds-gen` and `ellalgo` modules.

## 👀 See also

- [corr-solver-cpp](https://github.com/luk036/corr-solver-cpp)
- [Presentation Slides](https://luk036.github.io/cvx)

<!-- pyscaffold-notes -->

## 👉 Note

This project has been set up using PyScaffold 4.0.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.
