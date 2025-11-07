# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the Python package, including `celltype_gp_models.py` for deconvolution models, `evaluation.py` for metric reporting, and `preprocessing.py` for score generation. Add new model logic next to these modules and reuse shared helpers.
- `data/` stores `.npz` inputs, wide-format CSV outputs, and intermediate AnnData files; keep large raw datasets out of version control via `.gitignore` or symlinks.
- `notebooks/examples/` contains exploratory notebooks such as `demo_run.ipynb`; when notebooks introduce a pipeline, mirror the steps with CLI commands in README examples.

## Build, Test, and Development Commands
- `python src/celltype_gp_models.py --input data/spot_data_full.npz --method vectorized --save data/pred_result(vectorized).csv` runs the vectorized residual workflow end to end.
- `python src/evaluation.py` aligns predictions with `data/truth_output/` references and emits metric CSVs plus diagnostic plots.
- `python src/preprocessing.py` recomputes gene-program scores; execute inside an environment that already provides Scanpy, PyTorch, and sklearn to avoid dependency errors.

## Coding Style & Naming Conventions
- Follow 4-space indentation, snake_case identifiers, and bilingual docstrings or comments in modules that already mix English and Chinese notes.
- Prefer vectorized tensor math (NumPy/Torch) over Python loops; justify unavoidable loops with a brief inline comment.
- Notebooks should be saved with outputs cleared; scripts should expose an argparse-powered `__main__` entry point consistent with existing command-line patterns.

## Testing Guidelines
- Validate algorithm updates by rerunning `python src/evaluation.py` and comparing metrics (`PearsonR`, `F1`, `AUROC`) against baselines in `data/eval_summary_comparison.csv`.
- For unit-style checks, expand `test.py` or add a `tests/` module with `test_feature_expectedbehavior` naming and concise assertions.
- Document environment details (Python version, CUDA availability) in PR notes whenever performance or reproducibility depends on hardware.

## Commit & Pull Request Guidelines
- Repository history starts empty; write imperative commit subjects such as `Add vectorized residual baseline`, with wrapped body lines ≤72 characters and bullets for multi-step changes.
- Link issues or discussion threads in PR descriptions, summarise data dependencies, and attach before/after metric tables or plots for model-impacting work.
- Note any large files kept local in `data/` and provide reproduction commands so reviewers can run the same scripts.
