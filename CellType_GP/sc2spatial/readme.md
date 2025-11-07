# sc2spatial 项目说明

## 总体思路
- 以 CARD 模拟框架为核心，先在 R 中合并 split 单细胞数据、评估基因程序并准备抽样所需的元信息。
- 再根据预定义的层级标签与空间坐标，抽取真实单细胞表达来生成伪空间数据、真实比例与 cell–spot 映射。
- Python 端脚本/Notebook 将这些 CSV 输出进一步打包成 `npz`/`h5ad`，方便深度学习模型或 AnnData 流程直接加载。

## 代码与数据关系
| 组件 | 主要文件 | 输入 | 输出/作用 |
| --- | --- | --- | --- |
| 单细胞整合与特征评估 | `src/sc_merge_analysis.R` | `DATA/folder/simulations/split.scRNAseq.forSimu.RData` | Seurat 归一化与降维结果、marker/模块分数图 (`DATA/folder/*.png`)；为模拟提供 `ct.select`、marker gene set |
| 模拟空间数据 | `src/sc2sp_mapping.R` | 上述 `split.scRNAseq…`，`pattern_gp_label.RData`，`sim_MOB_location.RData` | 伪空间表达矩阵、真实细胞比例 (`pseudo_data/true_p/sample_random.csv`)，`cell_to_spot_mapping.csv`，`spot_to_cells.csv`，配对单细胞 (`sc_paired*.{RData,h5ad,mtx,csv}`) 等 |
| Python 数据打包 | `src/make_npz.py` | `DATA/sim_ctgp/*.csv` | `sim_sc2sp_dataset.npz`（整合表达、比例、映射等） |
| AnnData 导出 | `notebooks/VSVZ01_preprocessing.ipynb` | `pseudo_data.csv`, `true_p.csv` | `pseudo_data.h5ad` |

## 目录结构
```
sc2spatial/
├─ DATA/
│  ├─ folder/
│  │  ├─ simulations/              # split.scRNAseq、pattern_gp_label、sim_MOB_location 等原始资源
│  │  ├─ realdata/                 # 真实组织 ExpressionSet，可用于对照
│  │  └─ *.png                     # Seurat 降维与基因程序可视化输出
│  └─ sim_ctgp/
│     ├─ pseudo_data.csv           # 伪空间表达（gene × spot）
│     ├─ true_p.csv                # 每个 spot 的真实细胞类型比例
│     ├─ sample_random.csv         # Dirichlet 抽样得到的细胞计数矩阵
│     ├─ cell_to_spot_mapping.csv  # 逐细胞映射表，含 cell_id/spot_id/celltype
│     ├─ spot_to_cells.csv         # spot 聚合后的细胞列表
│     ├─ sc_paired*.{RData,h5ad,mtx,csv} # 与伪空间对应的单细胞表达与元数据
│     ├─ sp_coords.csv, coords_sc2sp.csv # 空间坐标、ID 映射
│     ├─ sp_fixed.h5ad, sp_score.npz     # 便于 downstream 模型的打包格式
│     ├─ sc_scores/, sp_scores/          # 基因程序在 sc/sp 上的得分图
│     ├─ truth_output/                   # `truth_result(wide|grouped).csv`, `truth_score.csv` 真值评估
│     └─ sim_sc2sp_dataset.npz           # make_npz.py 生成的一体化数据包
├─ notebooks/
│  └─ VSVZ01_preprocessing.ipynb
└─ src/
   ├─ sc_merge_analysis.R
   ├─ sc2sp_mapping.R
   └─ make_npz.py
```

## 操作顺序建议
1. **单细胞检查**：运行 `Rscript src/sc_merge_analysis.R`，确认 `ct.select`、marker 与模块打分是否符合实验设定，并查看 `DATA/folder/*.png` 输出。
2. **生成伪空间数据**：根据需要调整 `src/sc2sp_mapping.R` 中的 `ct.select`、`ntotal`、`imix`、`allow_reuse` 等参数后执行；产出的 CSV/RData 将写入 `DATA/sim_ctgp/`。
3. **导出模型输入**：
   - 需要 `npz` 时运行 `python src/make_npz.py`。
   - 需要 `h5ad` 时在 Notebook 中执行所有单元格，得到 `pseudo_data.h5ad`（同目录）。
4. **基准/真值评估**：使用 `DATA/sim_ctgp/truth_output/` 中的 `truth_result(wide).csv`、`truth_score.csv` 与 `cell_to_spot_mapping.csv` 对照解卷积或空间定位算法的预测。

## 数据流概览
1. `split.scRNAseq.forSimu.RData` → `sc_merge_analysis.R` 合并 + 基因程序打分。
2. `pattern_gp_label.RData` + `sim_MOB_location.RData` + 上一步结果 → `sc2sp_mapping.R` 生成 `spatial.pseudo`（表达 + 真实比例 + mapping）并持久化。
3. `pseudo_data.csv` 等 → `make_npz.py`/Notebook → `sim_sc2sp_dataset.npz`、`pseudo_data.h5ad`。
4. `truth_output/` 结合 `sp_scores/`、`sc_scores/` 用于可视化真实分布与算法预测差异。

## 其他补充
- `DATA/folder/realdata/` 中的多个组织 ExpressionSet 可以替换成新的 `pattern_gp_label`/`location` 以模拟不同结构；修改路径后重复第 2 步即可得到新数据集。
- `sc2sp_mapping.R` 中的 `mix1~3` 控制噪声比例（`imix` 取 0/1/2/3 对应 0/20/40/60%），适合按任务需求调节难度。
- 若需要记录更多属性（如模块得分、空间坐标）到 `npz/h5ad`，可在 `make_npz.py` 或 Notebook 内追加字段，保持一次性加载的便利。
- 所有脚本默认在 `/home/vs_theg/ST_program/CellType_GP/sc2spatial/` 下运行，如变更目录需同步修改脚本顶部的绝对路径。
