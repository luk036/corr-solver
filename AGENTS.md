# AGENTS.md - Agent Guidelines for corr-solver

## Build/Lint/Test Commands

### Testing
```bash
# Run all tests (coverage on by default)
pytest

# Run single test
pytest tests/test_corr_fn.py::test_lsq_corr_poly

# Run tests matching a pattern
pytest -k "corr_poly"

# With explicit coverage
pytest --cov corr_solver --cov-report term-missing

# Run via tox (isolated environment)
tox
```

### Linting & Formatting
```bash
# Run all pre-commit hooks (recommended before committing)
pre-commit run --all-files

# Individual tools
black .               # Format code (line length 188)
isort .               # Sort imports
flake8 .              # Lint (max_line_length=188, ignores E203/W503)
mypy src/             # Type check (Python 3.12 target)
```

### Build & Docs
```bash
tox -e build          # Build sdist + wheel
tox -e clean          # Remove build artifacts
tox -e docs           # Build HTML docs
tox -e doctests       # Run doctests
```

## Code Style Guidelines

### Project Structure
- Source code: `src/corr_solver/` (package name `corr_solver`; repo is `corr-solver`)
- Tests: `tests/` (test_corr_fn.py, test_corr_bspline_fn.py)
- Version managed via setuptools_scm (`no-guess-dev` scheme)

### Naming Conventions
- **Classes**: PascalCase (e.g., `QMIOracle`, `GMIOracle`); `mle_oracle` is an existing snake_case exception
- **Functions/Methods**: snake_case (e.g., `create_2d_sites`, `construct_poly_matrix`, `corr_poly`)
- **Type aliases**: Short PascalCase (e.g., `Arr = np.ndarray`, `Cut = Tuple[Arr, float]`)
- **Local constants**: lowercase (e.g., `sdkern`, `var`, `tau`)

### Type Hints
- **Required**: All function parameters and return types
- Use `typing` module: `Any`, `List`, `Optional`, `Tuple`
- NumPy arrays typed via alias: `Arr = np.ndarray`

### Docstrings
- **Primary style**: Sphinx/reStructuredText with `:param:`, `:type:`, `:return:`
- Long explanatory module docstrings describing algorithmic intent
- Doctest-style `Examples:` blocks where useful
- Test helper docstrings use Google-style `Arguments:`/`Returns:` (often `[summary]`)

### Imports
- **Order**: stdlib → third-party → local (PEP8, enforced by isort)
- Third-party example: `from lds_gen.lds import Halton`, `from scipy.spatial.distance import pdist, squareform`

### Error Handling
- **Minimal explicit error handling** — rely on natural exceptions
- Cutting-plane oracles return `(Cut, value)` tuples, not exceptions
- Use walrus operator for cut generation:
  ```python
  if cut := self.lmi.assess_feas(x):
      return cut, None
  ```
- `try/except` only for version detection in `__init__.py`

### Python Version
- Minimum: Python 3.10 (setup.cfg `python_requires`)
- CI tests on Python 3.10 and 3.11

### Configuration Files
- `setup.cfg`: package metadata, pytest options, flake8 config
- `.flake8`: formatter-friendly (ignores E1/E2/E3/E501/W1/W2/W3/W5)
- `.pre-commit-config.yaml`: pre-commit hooks
- `mypy.ini`: Python 3.12 target, ignores setuptools/matplotlib/ellalgo/lds_gen
- `pyproject.toml`: build system (setuptools_scm)
- `tox.ini`: task automation (test, build, clean, docs, doctests, publish)
- `.coveragerc`: branch coverage, excludes `__repr__`, debug, and assertion code

### Pre-commit Hooks
- trailing-whitespace
- check-added-large-files
- check-ast
- check-json / check-yaml / check-xml
- check-merge-conflict
- debug-statements
- end-of-file-fixer
- requirements-txt-fixer
- mixed-line-ending (auto-fix)
- isort
- black
- flake8

### Testing Patterns
- Framework: pytest with coverage (addopts: `--cov corr_solver --cov-report term-missing --verbose`)
- Float comparisons use `pytest.approx`
- Test functions typed `def test_...() -> None`
- Algorithm tests assert the `feasible` flag and iteration-count bounds

## Key Project Context

corr-solver fits correlation/error-correction polynomials to biased covariance matrices:
- **MLE oracle**: `mle_oracle` solves maximum-likelihood estimation subject to linear matrix inequality (LMI) constraints
- **LSQ variants**: `lsq_oracle` / `QMIOracle` provide least-squares alternatives
- **Site generation**: uses Halton low-discrepancy sequences to place 2D sites and build distance/correlation matrices
- **Solver backend**: ellipsoid method via `ellalgo` cutting-plane oracles (`cutting_plane_optim`, `bsearch`)
- **Key dependencies**: `numpy`, `scipy`, `ellalgo`, `lds_gen`
