# 🧬 CellType-GP: 空间转录组中细胞类型特异性基因程序建模软件设计文档

**作者：** [theg]  
**项目代号：** CellType-GP  
**版本：** v0.1  
**最后更新：** 2025-11-01

---

## 🧭 一、项目背景与科学动机

空间转录组技术（如 Visium、Xenium、MERFISH 等）为我们提供了组织空间中细胞类型和功能的分布信息。然而，低分辨率（如 Visium）平台的每个 spot 通常包含多种细胞类型，直接获得细胞类型特异的功能信息（Gene Program, GP）十分困难。

本项目旨在开发一个名为 **CellType-GP** 的软件框架，通过机器学习或深度学习建模，从低分辨率空间转录组数据中推断出每种细胞类型在每个 spot 上的基因程序活性。

### 研究目标

1. **模型功能一：基于细胞比例与基因集打分预测**  
   输入低分辨率空间转录组的细胞比例矩阵（spot × celltype）与基因集（Gene Programs），输出每个 spot 的 GP 得分矩阵。

2. **模型功能二：反卷积分解得到细胞类型特异 GP 矩阵**  
   根据每个 spot 的总 GP 得分，反卷积得到每种细胞类型在各个 spot 上的 GP 得分矩阵（celltype × GP × spot）。

3. **后续分析功能：**  
   基于细胞类型特异 GP 矩阵，可进行：
   - 细胞比例空间共定位分析  
   - ctGPs（celltype-specific GPs）共定位分析  
   - 空间功能互作网络推断

目前使用 **高分辨率的 Xenium 数据** 作为真实参考（truth），并使用同样的基因集在其对应的 **Visium 数据** 上进行打分，从而模拟模型的预测结果，并根据真实参考评价不同方法对细胞类型特异性基因程序的恢复能力。

---

## 🧩 二、总体框架

代码与数据布局（含脚本关系与运行顺序）：

```
CellType-GP/
├── celltype_gp_models.py         # 线性基线：加性贡献（向量化，主推）与 LOFO 重拟合
│   ├─ contribution_vectorized()  # Ridge 拟合后按细胞类型分解贡献，输出 Y_tps(T,P,S)
│   ├─ lofo_refit_residual()      # 逐细胞类型重拟合残差（慢，仅作对照）
│   └─ run_model()                # CLI 入口：读取 npz、运行方法、导出宽表 CSV
│
├── celltype_gp_deconvolution.py  # 神经网络原型（PyTorch）：基于图拉普拉斯的 ctGP 反卷积
│   ├─ 从 npz 读取 (Y,X,coords) 并构图 L
│   ├─ 调用外部包 `cell_program_deconvolution` 的 DeconvModel 与训练
│   ├─ 训练记录与 loss 曲线导出
│   └─ 将模型的 Y_tps 导出为宽表 CSV
│
├── train_utils.py                # 训练通用工具：早停、LR 调度、日志/曲线
├── grid_search_train_utils.py    # 并行网格调参：多组 (λ1, λ2, lr) 训练并导出宽表
│   └─ 需在 create_model() 返回你的 NN 模型实例
│
├── preprocessing.py              # 预处理：
│   ├─ 读取 Xenium/Visium，构建映射与对齐
│   ├─ 计算基因程序打分（可调用 score_gene_program.py）
│   ├─ 生成标准 npz: visium_score, spot_cluster_fraction_matrix, coords, names
│   └─ 导出中间 CSV（如 visium_program_scores.csv）
│
├── score_gene_program.py         # 基因程序打分（函数化，供预处理/验证复用）
├── compute_truth_score.py        # 从高分辨率数据（Xenium）生成“真值”打分或宽表
├── evaluation.py                 # 评估与可视化（数值拟合、散点图、分层统计）
│
├── examples/
│   ├─ demo_01_preprocessing.ipynb  # 预处理与数据构建示例
│   ├─ demo_02_run.ipynb            # 主演示：向量化加性贡献 + NN 方法与可视化
│   └─ deconvolution_test.ipynb     # 神经网络方法（反卷积）演示
└── README.md
```

运行逻辑与顺序（推荐）：
- 第一步 预处理与数据准备：使用 `preprocessing.py` 对齐 Xenium/Visium，生成 `DATA/spot_data_full.npz` 与派生 CSV。
- 第二步 模型分解（任选其一或同时运行）：
  - 线性基线：`celltype_gp_models.py --method vectorized` 生成 ctGP 宽表（主推）。
  - 神经网络：`celltype_gp_deconvolution.py` 或 `examples/deconvolution_test.ipynb` 中训练并导出宽表。
- 第三步 评估与可视化：使用 `evaluation.py` 或 `examples/demo_02_run.ipynb` 绘制散点、相关性排行、空间图等。


---

## 🚀 快速开始与命令（CLI）

- 运行向量化残差/加性贡献分解（默认）：
  `python celltype_gp_models.py --input DATA/spot_data_full.npz --method vectorized --save DATA/pred_result(vectorized).csv`
- 运行 LOFO 重拟合（较慢，用于对照）：
  `python celltype_gp_models.py --input DATA/spot_data_full.npz --method lofo_refit --save DATA/pred_result(lofo).csv`
- 评估预测并与真实参考对齐（输出指标与图）：
  `python evaluation.py`
- 重新计算基因程序打分与生成输入（需 Scanpy/PyTorch 环境）：
  `python preprocessing.py`

说明：命令默认在 `CellType_GP/CellType-GP/` 目录下运行，路径前缀 `DATA/` 指向同级目录 `CellType_GP/DATA/`。

---

## ⚙️ 三、数据结构与输入格式

| 变量名 | 形状 | 说明 |
|--------|------|------|
| `spot_cluster_fraction_matrix` | (S, T) | 每个 spot 中的细胞类型比例 |
| `visium_score` | (P, S) | 每个 spot 的 GP 得分 |
| `coords` | (S, 2) | spot 的空间坐标（本版本未使用） |
| `spot_names` | (S,) | spot 名称 |
| `celltype_names` | (T,) | 细胞类型名称 |
| `program_names` | (P,) | 基因程序名称 |

---

## 🧠 四、核心建模方法

本阶段 demo（demo_02）仅展示两类方法：

### 1) 加性贡献（向量化，主推）

- 思想：先用 Ridge 拟合 `Y≈X·β`；再将预测值按细胞类型分解为贡献项 `contrib_t(s,p) = X[s,t] · β[t,p]`，得到 `Y_tps (T,P,S)`。具备线性可解释性、向量化高效，是默认基线。
- 实现：`celltype_gp_models.py` 的 `contribution_vectorized()`，CLI 通过 `run_model()` 调用。
- 命令：`python celltype_gp_models.py --input DATA/spot_data_full.npz --method vectorized --save DATA/pred_result(vectorized).csv`

### 2) 神经网络反卷积（NN）

- 思想：以 `Y(obs)` 为重构目标，直接学习 `Y_tps` 使得 `sum_t Y_tps[t,:,:] ≈ Y`，可加入 L1 稀疏与空间平滑（基于坐标构图的拉普拉斯）。
- 实现与入口：
  - 脚本：`celltype_gp_deconvolution.py`（参数化训练、导出宽表、loss 曲线）。
  - 笔记本：`examples/deconvolution_test.ipynb`（更丰富的可视化与交互）。
  - 工具：`train_utils.py`（早停、LR 调度、日志/曲线），`grid_search_train_utils.py`（并行网格调参，需实现 `create_model()`）。
- 命令示例：`python celltype_gp_deconvolution.py --npz DATA/spot_data_full.npz --out DATA/nn_run --epochs 3000 --lr 1e-3 --lambda1 1e-4 --lambda2 1e-2`

注：旧小节（LOFO/Delta）已从演示主线中移除，仅保留在历史版本。

### 1️⃣ 残差法（LOFO 重拟合 Residual LOO；可选慢速）

**思想：**  
对每个细胞类型 t，将其从细胞比例矩阵中移除，拟合剩余类型与 GP 的关系，计算残差。该残差代表细胞类型 t 的特异性贡献。

**数学定义：**  
\[
Y_{tps} = Y - X_{-t}eta_{-t}
\]
其中：  
- \(Y\)：spot × program 矩阵  
- \(X\)：spot × celltype 比例矩阵  
- \(eta\)：每个 celltype 对 program 的线性系数（通过 Ridge 回归估计）

---

### 2️⃣ 贡献差值法（Delta LOO）

**思想：**  
通过全模型拟合与去除单个细胞类型拟合的预测差值，估算每个细胞类型对 GP 的贡献。

**数学定义：**  
\[
Y_{tps} = (Xeta) - (X_{-t}eta_{-t})
\]

与残差法不同，该方法更直接反映每个细胞类型的“加性贡献”。

---

### 3️⃣ 向量化实现（Additive Contribution Vectorized，默认）

**思想：**  
在残差法基础上，通过矩阵运算一次性计算所有细胞类型的贡献，大幅提高计算效率。

实现细节：先以 Ridge 拟合 Y≈X·β，随后将预测按 t 分解为贡献项 X[:,t]·β[t,:]，满足线性可加性。

命令示例：
- 运行模型（默认向量化贡献分解）：
  `python celltype_gp_models.py --input DATA/spot_data_full.npz --method vectorized --save DATA/pred_result(vectorized).csv`
- 运行 LOFO 重拟合（慢速）：
  `python celltype_gp_models.py --input DATA/spot_data_full.npz --method lofo_refit --save DATA/pred_result(lofo).csv`

预处理管线（参数化版）：
- 生成 npz 与评分输出：
  `python preprocessing.py --xenium DATA/xdata.h5 --visium DATA/vdata.h5 --fractions DATA/spot_cluster_fraction_matrix.csv --x2v-map DATA/xenium_to_visium_transcript_mapping.csv --outdir DATA`

---

## 📊 五、模型评估与可视化

### 1️⃣ 数值拟合指标（Continuous Fit Metrics）

| 指标 | 定义 | 说明 |
|------|------|------|
| **PearsonR** | 线性相关系数 | 衡量预测与真实得分之间的线性关系 |
| **SpearmanR** | 秩相关系数 | 衡量预测与真实得分的单调一致性 |
| **RMSE** | 均方根误差 | 衡量总体预测偏差 |
| **MAE** | 平均绝对误差 | 衡量预测偏离真实值的平均幅度 |

公式：  
\[
RMSE = \sqrt{rac{1}{n}\sum_i (y_i - \hat{y}_i)^2}, \quad MAE = rac{1}{n}\sum_i |y_i - \hat{y}_i|
\]

---

### 2️⃣ 可视化（demo_02 重点展示）

- 散点图：`truth` vs `prediction`（全量或按 CTGP），示例输出 `DATA/notebook_outputs/regression_scatter_top2_features.pdf`、`top5_features_scatter.pdf`。
- 相关性排行：Pearson/Spearman 条形图，示例 `DATA/notebook_outputs/pearson_corr_bar.png`、`regression_correlation_barplot.pdf`。
- 空间图：基于 `coords` 将指定 CTGP 的贡献 `Y_tps[t,p,:]` 着色可视化（热力/散点），示例 `DATA/notebook_outputs/spatial_Tumor_Invasive_Tumor_score.pdf`、`spatial_Stromal+Stromal_score.pdf`。
- 热图：
  - `T×P` 贡献强度（跨 S 聚合）或误差热图（`prediction - truth`），示例 `DATA/notebook_outputs/spatial_heatmaps.pdf`。
- 训练曲线（NN）：Total/Recon loss 随 epoch 变化，脚本版本自动保存 `loss_curve.png` 到运行输出目录；汇总例 `DATA/notebook_outputs/correlation_plot.pdf`。

---

## 🧮 六、结果解释

- 高 PearsonR & 低 RMSE → 数值拟合良好，整体误差较小。
- 相关性排行与热图 → 识别强/弱信号的 CT×GP 组合与系统性偏差。
- 空间图 → 定位功能热点与空间连续性特征；可与细胞比例或组织学背景联读。

---

## 📁 目录与数据约定（Repository Guidelines）

- 代码结构：
  - `CellType_GP/CellType-GP/`：Python 包与脚本入口；`celltype_gp_models.py`（反卷积分解）、`evaluation.py`（评估）、`preprocessing.py`（打分生成）。
  - `CellType_GP/DATA/`：输入 `.npz`、宽表 CSV 输出、中间 AnnData/H5；请勿版本化大文件，建议用 `.gitignore` 或符号链接本地化管理。
  - `CellType_GP/CellType-GP/examples/`：探索性笔记本（如 `demo_run.ipynb`）；README 中需提供对应 CLI 示例（已在“快速开始”给出）。
- 文件放置：新增模型/工具函数请与现有模块就近放置并复用公共工具，避免重复实现。

---

## 🛠️ 开发与编码规范

- 代码风格：4 空格缩进，变量/函数/文件使用 `snake_case`；优先采用向量化的 NumPy/PyTorch 张量运算。
- 性能说明：不可避免使用 Python 循环时，请在代码旁加入简短注释说明原因与复杂度考量。
- 文档与入口：脚本应提供 `if __name__ == "__main__":` 主入口与 `argparse` 参数；与双语文档风格保持一致（中英注释可并列）。
- Notebook 规范：提交前请清空输出，仅保留必要的文字与代码单元。

---

## ✅ 测试与基线对比

- 回归测试：修改算法后运行 `python evaluation.py`，并与 `DATA/truth_output/` 中的参考对齐，产出指标与图表。
- 关键指标：`PearsonR`、`SpearmanR`、`RMSE`、`MAE`；与 `DATA/eval_summary_comparison.csv` 基线对比。
- 轻量校验：为新增函数在 `test.py` 或 `tests/` 中补充断言/样例（命名如 `test_feature_expectedbehavior`）。
- 环境记录：结果依赖 GPU/CUDA 时，请在 PR 备注 Python 版本与 CUDA 可用性。

---

## 📌 提交与 PR 约定

- 提交信息：使用祈使句作为标题，如 “Add vectorized residual baseline”；正文每行 ≤ 72 字符，必要时用项目符号列出变更点。
- PR 描述：引用关联议题，概述数据依赖，并附“前后对比”指标表或图；注明 `DATA/` 新增的大文件应保持本地不入库。
- 复现说明：在描述中提供最小可复现实验命令（参考“快速开始”）。

---

## 🧩 七、未来扩展方向

| 模块 | 扩展方向 | 说明 |
|------|-----------|------|
| 模型层 | 深度学习（MLP、VAE） | 非线性建模，提升泛化能力 |
| 评估层 | 空间自相关分析 | 引入空间平滑性指标（Moran’s I） |
| 可视化层 | 可交互界面（Streamlit） | 动态查看各细胞类型 GP 分布 |

---

## 🧾 八、总结

CellType-GP 提供了一个高效且可解释的框架，用于从低分辨率空间转录组数据中推断细胞类型特异的基因程序活性。  
该框架以线性可解释模型与神经网络反卷积为核心，配合连续数值拟合指标与可视化分析，全面评估模型质量。

> “在空间维度中量化细胞类型功能，是连接单细胞与组织结构的关键一步。”
