# CODE_REVIEW_02

## 严重问题
- `CellType_GP/src/celltype_gp_models.py:191` 仍向 `lofo_refit_residual()` 传入 `positive` 参数，函数签名没有该参数，`method="lofo_refit"` 会立即抛出 `TypeError`。
- `CellType_GP/src/evaluation.py:200` 调用了未定义的 `plot_regression_scatter()`，脚本运行到此处直接崩溃。
- `CellType_GP/src/celltype_gp_deconvolution.py:29` 只把 `cell_program_deconvolution` 根目录加入 `sys.path`；当前包实际在 `cell_program_deconvolution/src`，导入模型必然失败。
- `cell_program_deconvolution/src/cell_program_deconvolution/deconvolution/train.py:8` 仍假定 `model.loss()` 返回两项，实际返回 `(total, recon, l1, smooth)`，训练循环触发 `ValueError: too many values to unpack`。
- `cell_program_deconvolution/src/cell_program_deconvolution/deconvolution/model.py:7-25` 未注册 `X`、`L` 为 buffer，`model.to("cuda")` 时会报设备不匹配；平滑项使用双重 for-loop（O(T·P·S²)），在真实数据上极慢。
- `cell_program_deconvolution/src/cell_program_deconvolution/deconvolution/Simulated Data Generation.py:38` 把演示数据写到 `/mnt/data/…`，新环境默认不存在该目录，导致脚本失败。
- `CellType_GP/scripts/test.py:12-22` 仍引用老路径 `/CellType_GP/src`，并调用已移除的 `method="delta"`；硬编码的 `/CellType_GP/data/…` 路径也和现目录不符。
- `CellType_GP/data/spot_cluster_fraction_matrix.csv\`` 带反引号的重复文件易误用，应整理或删除。

## 目录快照
- 根目录：`AGENTS.md`, `CODE_REVIEW.md`, `CODE_REVIEW_02.md`, `batch_install.py`, `CellType_GP/`, `cell_program_deconvolution/`, `visium_truth_correlation.csv`
- `CellType_GP/`
  - `src/`：`celltype_gp_models.py`, `celltype_gp_deconvolution.py`, `preprocessing.py`, `evaluation.py`, `score_gene_program.py`, `train_utils.py`, `grid_search_train_utils.py`, `compute_truth_score.py`
  - `data/`：`spot_data_full.npz`, `visium_program_scores.csv`, `xdata_processed.h5`, `vdata_processed.h5` 等输入与中间结果（含重复 `spot_cluster_fraction_matrix.csv\``）
  - `notebooks/`：`GSE243280_breastc_xenium.ipynb` 及 `examples/` 演示
  - `docs/`：项目文档、公式、PPT 草案、AGENTS
  - `sc2spatial/`:单细胞映射到空间spot的R 脚本`sc2sp_mapping.R`及相关数据集`/DATA`
  - `results/`：当前为空
- `cell_program_deconvolution/`
  - `src/cell_program_deconvolution/deconvolution/`：PyTorch 模型、图拉普拉斯、训练脚本、模拟数据
  - `notebooks/`：`01_run_pipeline.ipynb`, `Simulated Data Generation.ipynb`
  - `data/`：`example_inputs.npz`, `structured_inputs.npz`
  - `docs/`：README, AGENTS
  - `requirements.txt`

## 建议优先修复
1. 修正 `train_model`、`DeconvModel` 的返回值与设备处理，并向量化平滑项。
2. 更新所有导入路径与示例脚本，移除失效的 `method="delta"` 及老目录引用。
3. 为 `evaluation.py` 补上 `plot_regression_scatter()` 实现或删除调用，并清理重复文件。
4. 将 `/mnt/data/...` 输出改到项目内的可写目录，避免演示脚本直接报错。
