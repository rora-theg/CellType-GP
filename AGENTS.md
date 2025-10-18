# Repository Guidelines

## Project Structure & Module Organization
- `CellType_GP/CellType-GP/` contains the Python package, including `celltype_gp_models.py` for deconvolution models, `evaluation.py` for metric reporting, and `preprocessing.py` for score generation. Keep new model code colocated with these modules and reuse shared utilities.
- `CellType_GP/DATA/` stores `.npz` inputs, wide-format CSV outputs, and intermediate AnnData files; do not version large raw data and prefer symlinks or `.gitignore` for local artefacts.
- `CellType_GP/CellType-GP/examples/` holds exploratory notebooks such as `demo_run.ipynb`; mirror any scripted workflow in notebooks with CLI examples in the README.

## Build, Test, and Development Commands
- `python celltype_gp_models.py --input DATA/spot_data_full.npz --method vectorized --save DATA/pred_result(vectorized).csv` runs the vectorized residual pipeline end to end.
- `python evaluation.py` aligns predictions with `DATA/truth_output/` references and produces metric CSVs and histograms.
- `python preprocessing.py` recomputes gene-program scores; run inside a managed environment with Scanpy and PyTorch available to avoid dependency mismatches.

## Coding Style & Naming Conventions
- Use 4-space indentation, type-aware NumPy/PyTorch operations, and snake_case for variables, functions, and filenames. Keep bilingual docstrings or comments when extending modules that already use both English and Chinese notes.
- Vectorized tensor math is preferred over explicit Python loops; when loops are unavoidable, document the performance rationale in a short inline comment.
- Persist notebooks with cleared outputs; for scripted utilities, expose a `main` guard and argparse options consistent with existing entry points.

## Testing Guidelines
- Validate algorithm changes by re-running `python evaluation.py` and comparing key metrics (`PearsonR`, `F1`, `AUROC`) against prior baselines in `DATA/eval_summary_comparison.csv`.
- For new functions, add lightweight assertions or doctest-style checks inside `test.py` or a new `tests/` module; follow `test_feature_expectedbehavior` naming.
- Record environment details (Python version, CUDA availability) in PR notes when results depend on hardware acceleration.

## Commit & Pull Request Guidelines
- The Git history is currently empty; format new commit subjects in the imperative mood, e.g., `Add vectorized residual baseline`, and keep bodies under 72 characters per line with bullet lists for multi-step changes.
- Reference linked issues or discussion threads in PR descriptions, summarise data dependencies, and attach before/after metric tables or plots when model quality shifts.
- Mention any added files in `DATA/` that should remain local, and provide reproduction commands so reviewers can rerun the relevant scripts.
