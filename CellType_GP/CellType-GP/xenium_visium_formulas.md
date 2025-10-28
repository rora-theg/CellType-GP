# Xenium 真实值与 Visium 预分解得分的数学表达式

本文给出项目中两类核心量的数学定义：
- Xenium 真实值（truth，按 spot × celltype 聚合后的基因程序分数）
- Visium 分解前（预分解）的基因程序分数（用于模型输入的 `visium_score`）

为与当前实现一致，默认使用：
- 对表达矩阵进行 `log1p` 变换
- 基因程序分数为所含基因表达的算术平均
- 分数按样本轴做 Min-Max 归一化到 [0, 1]

---

## 符号约定
- 记基因程序集合为 \(\{G_p\}_{p=1}^P\)，其中 \(G_p\) 是程序 \(p\) 所含的基因集合。
- Xenium 单细胞索引为 \(i\)，其所属 Visium spot 记为 \(s(i)\)，细胞类型为 \(t(i)\)。
- Visium spot 索引为 \(s\)，细胞类型为 \(t\)。
- 表达量：
  - Xenium 单细胞基因表达为 \(x_{i,g}\)
  - Visium spot 级基因表达为 \(x^{\mathrm{vis}}_{s,g}\)
- 记 \(\mathrm{log1p}(x) = \log(1+x)\)。
- Min-Max 归一化：对一个向量 \(v\)，\(\mathrm{MinMax}(v)_k = \dfrac{v_k - \min(v)}{\max(v) - \min(v)}\)。

---

## 1) 基因程序打分（通用定义）
对任一对象（单细胞或 spot）在程序 \(p\) 下的原始分数：
\[
\tilde{s}_{\bullet,p} 
= \frac{1}{|G_p \cap \mathcal{V}|} \sum_{g\in G_p \cap \mathcal{V}} \mathrm{log1p}(x_{\bullet,g})
\]
其中 \(\mathcal{V}\) 为当前对象可用的基因集合；\(\bullet\) 表示该对象（Xenium 的细胞 \(i\) 或 Visium 的 spot \(s\)）。

归一化分数（默认 Min-Max，按对象轴进行）：
\[
 s^{\mathrm{norm}}_{\bullet,p} 
 = \mathrm{MinMax}\big(\{\tilde{s}_{\bullet,p}\}_{\bullet}\big)_{\bullet}
\]

对应实现：`CellType_GP/CellType-GP/score_gene_program.py` 中对 `scores` 的 `mean` 与 `MinMaxScaler` 处理。

---

## 2) Visium 分解前的分数（模型输入 Y）
对每个 Visium spot \(s\) 与程序 \(p\)：
\[
Y_{p,s} \equiv s^{\mathrm{norm}}_{s,p} 
= \mathrm{MinMax}\left(\left\{\frac{1}{|G_p \cap \mathcal{V}_{\mathrm{vis}}|}\sum_{g\in G_p \cap \mathcal{V}_{\mathrm{vis}}}\! \mathrm{log1p}\big(x^{\mathrm{vis}}_{s,g}\big)\right\}_{s}\right)_{s}
\]
这与生成 `npz` 中 `visium_score (P×S)` 的实现一致。

对应实现：`CellType_GP/CellType-GP/preprocessing.py` 中对 Visium 运行 `score_gene_programs` 并抽取 `*_score_norm` 形成 `visium_score`。

---

## 3) Xenium 真实值（truth）
先在 Xenium 单细胞上计算每个细胞的归一化程序分数 \(s^{\mathrm{norm}}_{i,p}\)。对给定的 Visium spot \(s\)、细胞类型 \(t\) 与程序 \(p\)，真实值定义为该组内细胞分数的平均：
\[
\mathrm{Truth}(s,t,p)
= \frac{1}{N_{s,t}} \sum_{i:\, s(i)=s,\, t(i)=t} s^{\mathrm{norm}}_{i,p}
\quad\text{其中}\quad N_{s,t}=\big|\{i: s(i)=s, t(i)=t\}\big|.
\]
当 \(N_{s,t}=0\) 时，该 \((s,t,p)\) 记为缺失（NaN）。

对应实现：`CellType_GP/CellType-GP/compute_truth_score.py` 中 `groupby([spot, celltype]).mean()` 对 `*_score_norm` 求均值，并在 `pivot` 后得到宽表。

---

## 备注
- 若选择不同归一化（如 z-score），可将上式中的 `MinMax` 替换为相应标准化算子；代码已支持 `norm_method = {minmax, zscore, none}`。
- 本文定义与 `preprocessing.py` 默认参数保持一致（`log1p=True, normalize=True, norm_method='minmax'`）。

