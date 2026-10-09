# Changelog

All notable changes to corr-solver will be documented in this file.

## Version 0.2 (2026-10-09)

### Features
- **Convex-concave procedure (CCP) MLE solver**: Promoted `cccp_mle_oracle.py` from experiments into the package and initialized the CCP from the free-MLE precision. (#df4ba81, #d92a506, #4887be4)
- **JIT-compiled oracle kernels**: Added `numba_oracles.py` and declared numba as an optional `numba` extra; the pure-Python and Numba oracles are unified behind one backend-injected implementation. (#c1184bd, #10899e7, #2ad305f)
- **Solver architecture**: Added a `Basis` strategy, solver layouts, a backend factory (`basis.py`, `backends.py`), shared types/protocols/kernels/math helpers (`types.py`, `protocols.py`, `kernels.py`, `math_utils.py`), and `SolverConfig` / `CoreResult` / `Assessment` value objects, with `SolverConfig` threaded through the fitting drivers. (#13c17f6, #acb4aaa, #f1c984d, #27990e8, #349c0c6, #2a957bd)
- **Kernel registry**: Unified the isotropic and anisotropic covariance generators behind one kernel registry. (#702d2cb)

### Performance
- **Vectorized gradient & outer buffer**: Reduced memory use with vectorized gradient and outer-buffer operations. (#a0c5ecb)

### Bug Fixes
- **B-spline knot domain**: Fixed the B-spline knot domain and the monotone coefficient constraint. (#5f3f5e3)
- **`python_requires` specifier**: Removed the stray quotes from `setup.cfg`. (#bcdb055)
- **mypy config**: Fixed the broken `mypy.ini`, simplified `__init__.py`, and removed stale artifacts. (#409be51)
- **CI repair**: Fixed broken `entry_points` and remaining skeleton imports. (#528c78d)
- **Tighter iteration bound**: Tightened the `lsq_corr_poly` iteration bound. (#7927bed)

### Testing & Code Quality
- **Numba oracle tests**: Added `tests/test_numba_oracles.py` and installed numba in CI so the JIT oracle tests run. (#c1184bd, #414eee1)
- **Solver tests**: Added `tests/test_cccp_mle.py`; rewired the oracles and tests onto the shared helpers. (#d92a506, #de712ce)
- **Complexity check**: Split the QMI kernel to satisfy the complexity check. (#aae66a6)

### Code Cleanup
- **Removed PyScaffold boilerplate**: Deleted dead code, dropped the Python-version compat shims, and cleaned the configs. (#ff424c6, #409be51)
- **Removed AI slop**: Stripped boilerplate from docstrings and comments. (#154f9d4)
- **Backend composition**: Dropped the Numba config subclasses in favour of backend composition; carried the default oracle/core on the basis and added an `OracleDecorator`. (#9b4958b, #a504703)
- **Experiment dedup**: Shared experiment helpers, dropped dead scripts, and parameterized the output. (#929532f, #de712ce)
- **Style**: Applied pre-commit across the tree and switched to double quotes. (#bd20506, #5ca9eba)

### Documentation
- **Non-parametric spatial-correlation paper**: Added `paper/spatial.md` with build config (crossref/latex/beamer YAML, preamble, `ieee.csl`, `.bib`, generated SVG/PDF figures), a limitations discussion, and reframed non-parametric motivation; added the reference papers. (#7cf409f, #1610a30, #755294b, #1327e6a, #b1949c2)
- **30-minute Beamer deck**: Added `paper/slides.md` and the generated PDF. (#490a146)
- **Sample-size sweeps & LSQ-vs-MLE experiments**: Added sweep scripts, a figure generator, and result figures. (#dbb7eb0, #72ba9a0, #7f4b64e, #7af6e13)
- **svgbob diagram**: Added a correlation-solver diagram to the module docstring. (#5245330)
- **AGENTS.md**: Added agent guidelines. (#93bc974)

### Build & CI
- **Updated GitHub Actions**. (#cf8a4a7)
