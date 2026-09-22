"""
Shared pytest fixtures for corr-solver.
"""

import numpy as np
import pytest

from corr_solver.corr_oracle import create_2d_isotropic, create_2d_sites


@pytest.fixture(scope="session")
def site() -> np.ndarray:
    """[summary]"""
    return create_2d_sites(5, 4)


@pytest.fixture(scope="session")
def Y(site: np.ndarray) -> np.ndarray:
    """[summary]"""
    return create_2d_isotropic(site, 3000)
