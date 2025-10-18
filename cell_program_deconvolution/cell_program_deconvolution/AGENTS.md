# Repository Guidelines

## Project Structure & Module Organization
Source code lives in `deconvolution/`, with `model.py`, `train.py`, and utilities such as `graph_utils.py` driving the core deconvolution pipeline. Notebooks for exploratory analysis and demos reside in `notebooks/`; start with `01_run_pipeline.ipynb` for an end-to-end walkthrough. Place input matrices (`Y.npz`, `X.npy`, etc.) and intermediate artifacts in `data/`. Keep large raw downloads outside the repo and symlink them into `data/` when needed.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create an isolated Python 3.9+ environment.
- `pip install -r requirements.txt`: install core dependencies (PyTorch, NumPy, Scanpy, scikit-learn).
- `jupyter notebook notebooks/01_run_pipeline.ipynb`: launch the reference workflow interactively.
- `python -m deconvolution.visualize --input data/results.npz`: example entrypoint for rendering outputs; adjust arguments to match your run artifacts.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and descriptive snake_case for functions, modules, and variables (e.g., `compute_laplacian`). Favor explicit imports from `deconvolution` modules and keep notebook cells reproducible by minimizing hidden state. Document new classes and complex functions with short NumPy-style docstrings outlining parameters and returns. Commit generated figures or checkpoints only if they are lightweight and essential for comprehension.

## Testing Guidelines
There is no formal test suite yet; add unit tests under `tests/` using `pytest` to validate tensor shapes, loss calculations, and graph construction helpers. Mirror module paths (e.g., `tests/test_graph_utils.py`). Include small synthetic fixtures in `tests/data/` to keep runtime under one minute. Run `pytest` before opening a pull request and attach relevant coverage notes when behavior changes materially.

## Commit & Pull Request Guidelines
Adopt Conventional Commit prefixes (`feat:`, `fix:`, `docs:`, etc.) followed by an imperative summary, keeping the subject under 72 characters. Reference notebook IDs or data assumptions in the body when applicable. Pull requests should outline the motivation, summarize functional changes, list verification steps (commands or screenshots), and mention any data prerequisites. Cross-link related issues or research notes so future contributors can trace decision context quickly.

## Data & Configuration Notes
Keep credentials and PHI out of the repository; rely on environment variables or `.env` files ignored by Git. When sharing notebooks, clear output cells or replace sensitive paths with placeholders. Document non-default hyperparameters and random seeds in `data/README.md` (create if absent) to make reproduction straightforward.
