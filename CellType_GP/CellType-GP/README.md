# 🧬 CellType-GP: 空间转录组中细胞类型特异性基因程序建模软件设计文档

**作者：** [theg]  
**项目代号：** CellType-GP  
**版本：** v0.1  
**最后更新：** 2025-10-16

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

```
📁 CellType-GP/
│
├── celltype_gp_models.py        # 核心模型文件（残差法、差值法、向量化实现）
├── evaluation.py                # 模型评估模块（拟合性 + 二分类指标）
├── preprocessing.py             # 打分、数据标准化与缺失值处理
├── examples/
│   └── demo_run.ipynb           # 使用示例
└── README.md
```

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

## 📊 五、模型评估体系

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

### 2️⃣ 二分类评估指标（Structural Binary Evaluation）

**背景：**  
由于并非每个 spot 都包含所有细胞类型，对应的真实打分可能为缺失（NaN）。这些 NaN 表示**结构性缺失（structural missingness）**，意味着该 spot 不含该细胞类型。

**策略：**  
将 NaN 视为 0，构建二分类任务：是否预测该 celltype–program 对存在。

| 指标 | 定义 | 说明 |
|------|------|------|
| **Precision (P)** | \( P = \frac{TP}{TP + FP} \) | 预测为正的样本中，真实为正的比例 |
| **Recall (R)** | \( R = \frac{TP}{TP + FN} \) | 真实为正的样本中，被正确预测的比例 |
| **F1 Score** | \( F1 = 2\frac{PR}{P + R} \) | 综合精度与召回率 |
| **AUROC / AUPR** | 面积指标 | 衡量模型区分正负样本的整体能力 |

---

## 🧮 六、结果解释

- **高 PearsonR & 低 RMSE** → 模型在数值层面拟合真实得分良好  
- **高 F1 / AUROC** → 模型在识别特定细胞类型存在与否方面准确  
- **coverage（覆盖率）** 表示真实信号非缺失比例  
- 结构性缺失的高比例说明某些细胞类型在该空间区域缺失，是生物学上合理的空间特征

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
该框架以**线性可解释模型**为基础，结合**连续值拟合指标**与**二分类结构性指标**，实现对模型性能的全面评估。

> “在空间维度中量化细胞类型功能，是连接单细胞与组织结构的关键一步。”
