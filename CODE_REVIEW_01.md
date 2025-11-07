# 代码与笔记本生物信息学审阅报告（CellType-GP）

> 目标：从生物信息学与工程可复现性角度，审阅仓库内 Python 代码与 Jupyter Notebook，指出潜在逻辑/实现问题并给出可操作修复建议。

本报告覆盖范围：
- 包 `CellType_GP/src/`（模型、预处理、评估等）
- 子项目 `cell_program_deconvolution/`（PyTorch 反卷积原型）
- 相关 notebooks（examples/ 与 notebooks/）


## 结论速览（优先级从高到低）
- 模型分解逻辑存在偏差与实现缺陷（sklearn/torch 混用、LOFO 概念与实现不一致）。
- 预处理与训练脚本普遍硬编码绝对路径与 `sys.path.append`，不利于复现与移植。
- `compute_truth_score` 存在交互式清洗、清洗结果未参与分组计算的问题。
- 基因集打分绘图依赖 UMAP/空间对象，缺少前置检查，容易在批处理失败。
- notebooks 多数未清空输出、含绝对路径；评审和 PR 指南未落地到当前版本。
- 深度模型与网格搜索脚本接口不一致，`create_model()` 未实现，测试脚本不可运行。
- 数据对齐与命名（ct/GP 顺序）存在潜在错位风险。


## 详细问题与修复建议

### 1) 线性分解模型（celltype_gp_models.py）
- 文件：`CellType_GP/src/celltype_gp_models.py:58`–`CellType_GP/src/celltype_gp_models.py:77`
- 问题1（类型/库混用）：`residual_vectorized` 中用 torch 张量直接传入 sklearn 的 `Ridge.fit`，可能得到 `dtype=object` 或直接报错。
  - 证据：`ridge.fit(X_scaled, Y.T)` 处 `X_scaled`、`Y` 在 `run_model` 中来源为 torch；函数内部也未显式 `.numpy()`。
- 问题2（方法学偏差）：函数名与 README 叙述指向“Residual LOO（去除单一细胞类型重新拟合）”，但实现使用了“全模型系数截取”，并未在每个 t 上重新拟合 β_-t。对于相关性较强的细胞类型，系数会显著变化，导致归因偏差。
- 问题3（接口/文档不一致）：`run_model(..., method="delta")` 会调用 `delta_loo`，但该函数被注释删除；`test.py` 仍在调用 delta。
- 问题4（空间变量未用）：读入 `coords` 未使用，建议删除或用于空间正则版本。

修复建议：
- 最小变更（可立即落地）：
  - 将 sklearn 拟合使用 numpy，随后再转回 torch；并提供“可解释加性贡献”作为默认输出，避免“伪 LOFO”。
  - 具体实现（思路）：
    - 拟合：`ridge.fit(X_scaled_np, Y_np.T)` 得到 `coefs_np (T,P)`。
    - 可解释贡献分解：`contrib = np.einsum('st,tp->tps', X_scaled_np, coefs_np)`，返回形状 `(T,P,S)`；其按 t 求和即为 `Y_hat`，与线性模型可加性一致，且避免两次相近预测相减带来的数值不稳定。
  - 将 `method` 选项改为 `vectorized`（默认）与 `lofo_refit`（慢速、逐 t 重拟合）两种；在 README 中注明差异与适用场景。
- 正统 LOFO（可选）：逐 t 重新拟合 β_-t（for t in T），得到 `Y - X_{-t}β_{-t}`，准确但 O(T) 次拟合开销大；可仅在小 T 或评估时启用。
- 对负值：若后续“存在性”使用阈值判断，建议在导出前对 `Y_tps` 做简单截断或标准化（如按程序内 z-score、或最小值截断到 0），并在 `evaluation.py` 中记录阈值/变换。


### 2) 预处理流程（preprocessing.py）
- 文件：`CellType_GP/src/preprocessing.py`
- 问题1（硬编码路径/切换 CWD）：多处绝对路径和 `os.chdir`，导致不可移植且易污染环境。
- 问题2（导入路径注入）：`sys.path.append` 指向本机绝对路径，应改为包内相对导入或脚本参数。
- 问题3（spot 对齐潜在异常）：
  - `adata_v = adata_v[target_spots]` 在存在缺失 spot 时会 `KeyError`。
  - 建议使用交集索引后的“重排序”写法：`order = adata_v.obs_names.intersection(target_spots); adata_v = adata_v[order].copy()` 并检查覆盖率。
- 问题4（细胞类型名称硬编码）：`celltype_names` 固定列表可能与 `spot_cluster_fraction_matrix` 列顺序不一致，引发下游错位。
- 问题5（真值依赖）：`compute_truth_score` 假设 `broad_annotation` 等列存在，需在 README/CLI 参数中显式约束。

修复建议：
- 增加 CLI：`--xenium`, `--visium`, `--fractions`, `--outdir`，全部路径参数化；移除 `os.chdir` 与绝对路径。
- 从 `spot_cluster_fraction_matrix.columns` 派生 `celltype_names`；从 `score_gene_programs` 的返回或 `gene_sets_to_score` 键派生 `program_names`，保证顺序一致。
- 对齐顺序：按 `fractions.index` 重新排序 visium adata，并打印“覆盖率/丢失数”；若存在丢失，保存一份差异列表供排查。


### 3) 真值计算（compute_truth_score.py）
- 文件：`CellType_GP/src/compute_truth_score.py`
- 问题1（交互式清洗）：`clean_obs_data()` 默认交互输入，批处理会阻塞；
- 问题2（清洗结果未使用）：`compute_truth_score()` 调用 `df_clean = clean_obs_data(adata)`，但随后的 `compute_group_means(adata.obs)` 并未使用该清洗后的 DataFrame。

修复建议：
- 让 `compute_truth_score(adata, drop_columns=None)` 将 `drop_columns` 传递给 `clean_obs_data` 并在分组时使用清洗后的 `df_clean`：
  - `truth_result = compute_group_means(df_clean)`
  - 或将 `adata.obs` 替换为清洗结果再计算。
- 将 `broad_annotation`/`transcript_level_visium_barcode` 作为参数，并在缺失时给出友好错误信息与可选映射表路径参数。


### 4) 基因集打分与可视化（score_gene_program.py）
- 文件：`CellType_GP/src/score_gene_program.py`
- 问题1（UMAP 依赖）：若未预先计算 UMAP（`adata.obsm['X_umap']`），`sc.pl.umap` 可能报错；
- 问题2（空间绘图依赖）：squidpy 绘图依赖 `adata.uns['spatial']`/`library_id`，不同平台字段名差异大，需前置检查；
- 问题3（归一化策略）：对每个基因集做 MinMaxScaler 到 [0,1]，跨数据集可比性和“存在性阈值=0”假设并不稳健；
- 问题4（输出管理）：绘图始终保存，notebook 内批量运行易产生大量静态图，建议按需开关。

修复建议：
- 增加安全检查：若无 UMAP 则跳过 UMAP 图；若无 `spatial` 元数据则跳过空间图，并提示如何准备。
- 提供 `normalize={'none'|'zscore'|'minmax'}` 选项；用于下游 presence 阈值需在 `evaluation.py` 中一致记录。
- 返回被实际使用的 `score_cols` 顺序，供 `preprocessing.py` 精确使用。


### 5) 评估指标（evaluation.py）
- 文件：`CellType_GP/src/evaluation.py`
- 当前实现总体合理：
  - 对齐行列；
  - 连续指标（Pearson/Spearman/MAE/RMSE）与存在性指标（混淆矩阵 + ROC/AUPR）。
- 细节建议：
  - 默认 `presence_threshold=0.0` 对含负值预测不稳健，建议支持基于分位数/Youden 指标的自动阈值，或在导出前进行非负化处理（并记录变换）。
  - 散点图内联生成 `merged_data` 重复计算，可直接复用 `compute_regression_metrics` 返回的 `merged`。


### 6) 深度反卷积原型（cell_program_deconvolution/*）
- 模型文件：`cell_program_deconvolution/src/cell_program_deconvolution/deconvolution/model.py`
  - 逻辑正确（Y ≈ Σ_t X[:,t]·Y_tps[t,:,:]），但空间平滑损失逐 (t,p) 双重 for 循环，计算效率低。
  - 建议向量化：将 `Y_tps` reshape 为 `(T*P, S)` 后，统一计算 `trace(Y L Y^T)` 形式（或按批次）。
  - 可选加入非负约束（如 softplus）提升可解释性。
- 训练与可视化：`train.py`/`visualize.py` 基本 OK。
- 集成脚本：`CellType_GP/src/celltype_gp_deconvolution.py`
  - 问题1：绝对路径与 `sys.path.append`；
  - 问题2：`celltype_names`/`program_names` 被重新硬编码，易与 `.npz` 中顺序不一致导致错位；
  - 问题3：始终 `plt.show()`，批处理流程可能阻塞。

修复建议：
- 参数化所有路径，直接读取 `.npz` 中的 `spot_names/celltype_names/program_names`；
- 统一列名拼接逻辑与 `Ytps_to_wide_df` 保持一致；
- 增加 `--no-show` 选项，仅保存图。


### 7) 网格搜索工具（grid_search_train_utils.py）
- 文件：`CellType_GP/src/grid_search_train_utils.py`
- 问题：`create_model()` 未实现，脚本不可运行；
- 建议：在 README 中给出示例实现或将其改为通过入口点/模块路径字符串导入（例如 `--model-factory mypkg.mymod:create_model`）。
- 细节：保存 `meta.json` 已包含关键信息，建议补充 `seed/device/CTGP 形状`。


### 8) 测试脚本（test.py）
- 文件：`CellType_GP/test.py`
- 问题：
  - 绝对路径；
  - 调用 `method="delta"`，但 `delta_loo` 已移除；
  - 缺少 `import numpy as np` 等依赖。
- 修复：改为 `method="vectorized"`，参数化路径；加入基本断言（例如列名/形状对齐）。


### 9) Notebooks 审阅
- 文件：
  - `CellType_GP/notebooks/examples/demo_run.ipynb`
  - `CellType_GP/notebooks/examples/demo_01_preprocessing.ipynb`
  - `CellType_GP/notebooks/examples/deconvolution_test.ipynb`
  - `CellType_GP/GSE243280_breastc_xenium.ipynb`
  - `cell_program_deconvolution/notebooks/*.ipynb`
- 发现：
  - 多数含输出（未清空），与仓库约定不符。
  - 存在绝对路径（例如 `.../ST_program/...`、`/mnt/data/...`）。
- 建议：
  - 清空输出、使用相对路径与环境变量（或 `.env`/argparse）注入数据路径；
  - 在 README 中提供 CLI 对应的等价 Notebook 片段，确保二者一致；
  - 若包含大图/中间结果，建议输出到 `data/` 并 `.gitignore`，Notebook 仅保留代码与关键可视结论（小体积 PNG）。


### 10) 数据与版本控制
- `CellType_GP/results/**/best_model.pt` 等二进制文件已入库，不符合“data 大文件不入库”的约定；
- 建议：将 `results/**`, `data/**/*.h5`, `*.pt`, `*.npz`、大图等加入 `.gitignore`，并在 PR 中说明本地生成内容与重现命令。


## 生物学逻辑校核要点
- ct-GP 分解的线性假设（加性）在 `vectorized` 版本中应通过“贡献分解”（X·β 的逐 t 项）明确体现；当前“伪 LOFO”会因 X 各列相关性导致不稳定归因。
- 真值构建采用 Xenium 单细胞在 `spot×broad_annotation` 上取均值的策略合理，但应：
  - 明确 `broad_annotation` 的来源与标准化；
  - 对于细胞计数极小（n<k）的 group 可考虑最小样本过滤或加权；
  - 记录 `coverage`（非缺失比例）并在评估中作为分层指标。
- 基因集来源与命名：当前 DCIS 列表中包含 `DCIS1:HPX`（疑似带注释的基因名），建议统一为有效基因符号并记录版本（HGNC/Ensembl 与平台 panel 的交集），避免跨平台评分丢失。
- 评分归一化策略（min-max 到 [0,1]）会影响不同数据批/平台的可比性，建议在方法学上固定一种跨数据集稳健的方案（如 log1p 均值后 z-score by spot 或 by program），并在评估阈值上保持一致。


## 建议的修复落地顺序（可 1–2 次 PR 完成）
1) 修复 `celltype_gp_models.py`：numpy/sklearn 兼容、默认贡献分解、移除/重命名 delta；补充 README 方法学说明。
2) 参数化 `preprocessing.py` 与 `celltype_gp_deconvolution.py` 路径、对齐顺序、派生名称；移除 `sys.path.append` 与 `os.chdir`。
3) `compute_truth_score.py`：取消交互、使用清洗后的 df；阐明关键列来源并参数化。
4) `score_gene_program.py`：前置检查 + 可配置归一化；绘图按需开关。
5) 清理 notebooks 输出与绝对路径；将大文件加入 `.gitignore`。
6) `grid_search_train_utils.py` 给出 `create_model()` 示例或以入口点加载；`test.py` 修复为可跑。


## 附：关键位置参考
- `CellType_GP/src/celltype_gp_models.py:58` — sklearn/torch 混用与“伪 LOFO”实现
- `CellType_GP/src/preprocessing.py:1` — 绝对路径/`sys.path.append`/`os.chdir`
- `CellType_GP/src/compute_truth_score.py:65` — 交互清洗未参与分组
- `CellType_GP/src/score_gene_program.py:63` — UMAP/Spatial 绘图前置检查缺失
- `CellType_GP/src/evaluation.py:189` — 评估流程与可改进点
- `CellType_GP/src/celltype_gp_deconvolution.py:1` — 绝对路径与重写名称
- `CellType_GP/src/grid_search_train_utils.py:1` — `create_model()` 未实现
- `CellType_GP/test.py:1` — 不可运行（delta 方法）
- `cell_program_deconvolution/src/cell_program_deconvolution/deconvolution/model.py:1` — 空间平滑可向量化


—— 审阅完毕。如需，我可以按上述顺序提交小步 PR（先模型与预处理，再真值与评估）。
