"""Correlation Solver

.. svgbob::
   :align: center

        Input ──► Correlate ──► Output
                   │     ▲
                   │     │
                   ▼     │
                Feedback Loop
"""

from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

try:
    dist_name = "corr-solver"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
