# CellType–GeneProgram (CTGP) 报告大纲（逐页细化 + 可复现步骤）

本大纲用于直接制作PPT，每一页提供：页标题、要点/图表、需要准备的素材、可复现命令、注意事项。按顺序完成即可得到可讲、可复现的完整报告。

---

## 1. 标题页（Title）
- 标题：CellType–GeneProgram 分解与评估（Xenium/Visium）
- 副标题：向量化加性分解、LOFO 残差、PyTorch 反卷积原型
- 作者、单位、日期
- 素材：无

备注（Speaker Notes）
- 简要描述动机：在空间转录组中，解析“细胞类型 × 基因程序”的空间贡献图谱。

---

## 2. 研究目标与贡献（Background & Goals）
- 要点
  - 目标：将 Visium 的基因程序分数，按细胞类型贡献拆解为 Y_tps(t,p,s)。
  - 贡献：
    - 标准化数据管线（Xenium 真值 + Visium 打分 + fractions 对齐）。
    - 两种可解释/对照方法：向量化加性分解、LOFO 残差；以及图先验的反卷积原型。
    - 统一评估框架（拟合与存在性两条指标）。
- 素材：无

备注
- 突出“可复现”“可解释”“可扩展（加入空间先验）”。

---

## 3. 记号与数学设定（Notation）
- 记号
  - S：spot 数；T：细胞类型数；P：基因程序数。
  - `X ∈ R^{S×T}`：spot×celltype 比例矩阵，X[s,t] ≥ 0，∑_t X[s,t] ≈ 1。
  - `Y_obs ∈ R^{P×S}`：Visium 观测的程序分数矩阵（每列一个 spot）。
  - `Y_tps ∈ R^{T×P×S}`：待求的“细胞类型×程序×空间”贡献张量。
  - 空间图 `G=(V,E)`，|V|=S；拉普拉斯 `L ∈ R^{S×S}`。
- 目标
  - 分解关系：对每个程序 p、spot s，有
    - `Y_obs[p,s] ≈ ∑_{t=1}^T X[s,t] · Y_tps[t,p,s]`。
  - 输出宽表：按列拼接 `t+p` 形成 S×(T·P) 的表。

---

## 4. 数据与预处理概览（Data & Preprocessing）
- 要点
  - 输入：Xenium `.h5ad`、Visium `.h5ad`、`spot×celltype fractions.csv`、Xenium→Visium 条形码映射。
  - 输出：标准 `spot_data_full.npz`（供所有模型统一使用）。字段说明：
    - `visium_score (P×S)`、`spot_cluster_fraction_matrix (S×T)`、`coords (S×2)`、`spot_names`、`celltype_names`、`program_names`。
- 图表/示意
  - 流程图：Xenium 真值 → Visium 打分 → 对齐 → NPZ。
- 命令（生成标准 NPZ）
  - `CellType_GP/src/preprocessing.py:1`
  - 运行：
    ```bash
    python CellType_GP/src/preprocessing.py \
      --xenium /path/to/xenium.h5ad \
      --visium /path/to/visium.h5ad \
      --fractions CellType_GP/data/spot_cluster_fraction_matrix.csv \
      --x2v-map /path/to/x2v_mapping.csv \
      --outdir CellType_GP/data
    ```
- 产出文件
  - `CellType_GP/data/spot_data_full.npz`
  - `CellType_GP/data/truth_output/` 下若干 CSV（真值宽表等）
- 注意
  - 需安装 Scanpy/PyTorch 环境；Xenium obs 中需存在匹配的 cell id 列（脚本会自动检测）。

---

## 5. 基因项目录与打分（Gene Programs）
- 要点
  - 预置 4 组示例基因集：`DCIS_1/2`, `Prolif_Invasive`, `Invasive_Tumor`。
  - 评分产物：`*_score_norm` 列写入 AnnData.obs，并在生成 NPZ 时抽取为 `visium_score`。
- 参考代码
  - `CellType_GP/src/preprocessing.py:56`
  - `CellType_GP/src/score_gene_program.py:1`
- 检查点
  - 运行预处理后，确认 `CellType_GP/data/visium_scores/` 下有导出文件；`spot_data_full.npz` 中包含 `program_names`。

---

## 6. 真值构建（Truth from Xenium）
- 要点
  - 基于 Xenium→Visium 映射，将 Xenium 细胞聚合为 Visium spot；按 `broad_annotation` 分组求均值得到真值宽表。
- 参考代码
  - `CellType_GP/src/preprocessing.py:78`
- 产出
  - `CellType_GP/data/truth_output/` 目录下的宽表 CSV（用于评估脚本的 --truth）
- 注意
  - 脚本会打印映射缺失与过滤数量，记录到实验笔记。
- 数学表达（一次性讲清楚）
  - 设 Xenium 单细胞的程序分数为 `g_{i,p}`，细胞类型标签为 `ct_i ∈ {1..T}`，映射到的 Visium spot 为 `m(i)=s`。
  - 对每个 `(s,t,p)`，真值定义为该 spot 内、该细胞类型的单细胞程序均值：
    - `Truth[s,(t,p)] = mean_{i: m(i)=s, ct_i=t} g_{i,p}`（若集合为空则记为缺失）。
  - 对应宽表即以 `(t,p)` 为列键的 `S×(T·P)` 矩阵，用于评估。

---

## 7. 方法一：向量化残差/加性分解（Vectorized Additive）
- 要点（讲原理）
  - 拟合 Y ≈ X·β（Ridge，多目标），贡献分解 Y_tps(t,p,s) = X[s,t]·β[t,p]。
  - 采用 CLR 处理比例共线性，可选列标准化稳健拟合；计算全向量化，效率高。
- 参考代码
  - `CellType_GP/src/celltype_gp_models.py:39` (`contribution_vectorized`)
- 运行命令
  - `CellType_GP/src/celltype_gp_models.py:167`
  - ```bash
    python CellType_GP/src/celltype_gp_models.py \
      --input CellType_GP/data/spot_data_full.npz \
      --method vectorized \
      --save CellType_GP/data/pred_vectorized.csv \
      --alpha 0.1
    ```
- 产出
  - `CellType_GP/data/pred_vectorized.csv`（宽表：行=spot，列=ct+program）
- 注意
  - 默认启用 CLR；如需禁用标准化，添加 `--no-standardize`。
- 数学表达与过程
  - 预处理：对 `X` 可做 CLR：`X_clr[s,t] = log( X[s,t] / (∏_{t'} X[s,t'])^{1/T} )`，并按列标准化得到 `X_in`。
  - 多目标 Ridge：
    - 记 `Y = Y_obs^T ∈ R^{S×P}`，`β ∈ R^{T×P}`。
    - `β* = argmin_β || X_in · β − Y ||_F^2 + α ||β||_F^2`。
  - 向量化贡献分解：
    - `Ŷ(p,s) = ∑_t X_in[s,t] · β*[t,p]`；
    - `Y_tps[t,p,s] = X_in[s,t] · β*[t,p]`（einsum 实现为 `'st,tp->tps'`）。
  - 可选对照（LOFO 残差）：
    - `Y_tps[t,:,:] = Y_obs − X_{−t} · β_{−t}*`，其中 `β_{−t}*` 由 `X` 去掉第 t 列后重拟合得到。

---

## 8. 方法二：神经网络反卷积（Graph-Regularized NN；命令与产物）
- 要点（讲直觉与开销）
  - 直接学习 `Y_tps ∈ R^{T×P×S}`，使得 `Y_pred[p,s] = ∑_t X[s,t] · Y_tps[t,p,s]` 逼近 `Y_obs[p,s]`；
  - 同时在空间图上对每个 `(t,p)` 的向量 `Y_tps[t,p,:]` 施加拉普拉斯平滑。
- 参考代码
  - `cell_program_deconvolution/deconvolution/model.py:1`
- 运行命令/工具与 Notebook
  - 运行脚本：`CellType_GP/src/celltype_gp_deconvolution.py:1`
  - 训练工具：
    - 基础：`cell_program_deconvolution/deconvolution/train.py`
    - 优化版工具：`CellType_GP/src/train_utils.py`（加速/监控封装，可在后续整合使用）
  - 交互式 Notebook：`CellType_GP/notebooks/examples/deconvolution_test.ipynb`
  - 命令示例：
    ```bash
    python CellType_GP/src/celltype_gp_deconvolution.py \
      --npz CellType_GP/data/spot_data_full.npz \
      --out CellType_GP/data/deconv/ \
      --epochs 3000 --k 6 --lr 1e-3 --lambda1 1e-4 --lambda2 1e-2 --no-show
    ```
- 产出
  - `CellType_GP/data/deconv/ctgp_deconv(wide).csv`
- 注意
  - 关注收敛与平滑强度 λ2；可调 l1 以促稀疏。
- 数学目标与正则
  - 预测：`Y_pred = X^T ⊗ I_P` 与 `Y_tps` 的张量收缩，代码为 `einsum('st,tps->ps')`：
    - `Y_pred[p,s] = ∑_{t=1}^T X[s,t] · Y_tps[t,p,s]`。
  - 目标函数：
    - `L_total = ||Y_pred − Y_obs||_F^2 + λ1 · ||Y_tps||_1 + λ2 · ∑_{t,p} (y_{t,p})^T L y_{t,p}`，
    - 其中 `y_{t,p} = Y_tps[t,p,:] ∈ R^S`，`L` 为图拉普拉斯，平滑项鼓励空间相邻 spot 数值接近。
  - 训练：最小化 `L_total` 学得 `Y_tps`，再导出宽表。

---

## 9. 评估方法与实现（Metrics & Implementation）
- 要点
  - 拟合类：Pearson、Spearman、MAE、RMSE（整体与 per-feature）。
  - 存在性类：F1、Accuracy、ROC-AUC、Average Precision（阈值可调）。
  - 自动对齐 truth/pred 的行列交集，确保一一对应。
- 参考代码
  - `CellType_GP/src/evaluation.py:1`
- 运行命令（以向量化结果为例）
  - ```bash
    python CellType_GP/src/evaluation.py \
      --truth CellType_GP/data/truth_output/<truth_wide>.csv \
      --prediction CellType_GP/data/pred_vectorized.csv \
      --method-name vectorized \
      --presence-threshold 0.0 \
      --summary-path CellType_GP/data/eval_summary_vectorized.csv \
      --per-feature-regression CellType_GP/data/per_feature_reg_vectorized.csv \
      --per-feature-presence CellType_GP/data/per_feature_presence_vectorized.csv \
      --no-show
    ```
- 产出
  - 汇总：`CellType_GP/data/eval_summary_vectorized.csv`
  - 明细：`CellType_GP/data/per_feature_reg_vectorized.csv`、`CellType_GP/data/per_feature_presence_vectorized.csv`
- 注意
  - `--presence-threshold` 建议扫值做敏感性分析（见第 12 页）。

---

## 10. 结果页一：整体指标（Overview）
- 要点展示
  - 表格：Vectorized vs LOFO vs Deconv（pearson、spearman、MAE、RMSE、F1、ROC-AUC、AP、n_pairs）。
  - 结论句：向量化在速度与稳定性上表现更优，指标与 LOFO 接近/更好；Deconv 原型在若干程序上可视化更平滑。
- 素材准备
  - 汇总 CSV：`CellType_GP/data/eval_summary_*.csv`（可手工合并为一张对比表）。
- 可选图
  - 各指标柱状对比图。

---

## 11. 结果页二：代表性 CT×Program 可视化
- 要点展示
  - 选择 2–3 个代表性组合（一个表现好、一个一般、一个较差），展示：
    - 预测 vs 真值散点；
    - 空间热图（若使用 Deconv，可对比更平滑的空间分布）。
- 素材准备
  - 来自评估合并数据的散点；空间热图可用 `cell_program_deconvolution` 的可视化工具或自制 matplotlib。
- 操作提示
  - 选择 `per_feature_*` 中排行前/后的 CT+GP 作为示例。

---

## 12. 消融：阈值/正则/标准化影响（Ablations）
- 要点
  - `presence-threshold` 扫描：F1/PR/AUC 随阈值变化曲线。
  - `alpha`（Ridge）、是否启用列标准化、是否启用 CLR 的影响。
- 操作步骤
  - 扫描阈值：循环调用 `evaluation.py`，不同 `--presence-threshold`，将结果汇总到一张表。
  - 调参：重复运行 `celltype_gp_models.py` 修改 `--alpha` 与 `--no-standardize`，记录指标变化。
- 输出
  - 折线图：F1/ROC-AUC vs 阈值；表格：关键超参对整体 Pearson/MAE 的影响。

---

## 13. 误差分析（Error Analysis）
- 要点
  - 从 `per_feature_*` 明细找出偏差大的 CT+GP：
    - 共线性或稀有细胞型导致的误差；
    - 某些程序可能跨细胞型表达，贡献分解模糊；
    - fractions 估计误差的传导。
- 素材
  - `CellType_GP/data/per_feature_reg_*.csv`、`CellType_GP/data/per_feature_presence_*.csv`
- 建议图
  - 误差分布直方图，挑选异常点做个案图。

---

## 14. 讨论：优势、局限与改进方向（Discussion）
- 要点
  - 优势：可解释、可复现、计算高效；评估覆盖拟合与存在性两条线；管线标准化。
- 局限：依赖 fractions 质量；基因集迁移性；Deconv 仍原型阶段需系统调参。
- 未来：更强空间/结构先验、更细粒度基因程序、端到端弱监督学习。

---

## 15. 复现指引与环境（Repro & Env）
- 要点
  - 一键命令清单（建议顺序）：
    1) 数据与预处理（见第 4 节）
    2) 向量化分解（见第 7 节）
    3) 神经网络反卷积（见第 8 节，或用 Notebook）
    4) 评估与导出（见第 9 节）
  - 环境：Python 版本、Scanpy、scikit-learn、PyTorch（CUDA 可选）。
- 建议写入 README 的示例命令
  - 生成 NPZ、运行三种方法、评估与汇总的完整命令片段（已在前文给出）。
- 注意
  - 大型原始数据不入库；`CellType_GP/data/` 下仅放结果或小型示例，原始路径用本地符号链接或 `.gitignore`。

---

## 16. 结论与致谢（Conclusion）
- 要点
  - 方法闭环：标准化数据 → 可解释分解/原型 → 统一评估 → 诊断优化。
  - 实际观测：向量化方法作为稳健基线；图先验展现可视化潜力。
  - 下一步计划与协作邀请。

---

## 附录 A：文件清单与关键入口
- 代码入口
  - `CellType_GP/src/preprocessing.py:1`
  - `CellType_GP/src/celltype_gp_models.py:167`
  - `CellType_GP/src/evaluation.py:1`
  - `CellType_GP/src/celltype_gp_deconvolution.py:1`
- 数据与结果
  - `CellType_GP/data/spot_data_full.npz`
  - `CellType_GP/data/truth_output/`
  - `CellType_GP/data/pred_vectorized.csv`
  - `CellType_GP/data/pred_lofo.csv`
  - `CellType_GP/data/deconv/ctgp_deconv(wide).csv`
  - 训练工具与 Notebook：`CellType_GP/src/train_utils.py`，`CellType_GP/notebooks/examples/deconvolution_test.ipynb`

---

## 附录 B：常见问答（Q&A）
- 为什么使用 CLR/标准化？
  - 解决组成数据共线性与量纲问题，提升 Ridge 拟合稳定性。
- 阈值如何选择？
  - 扫描阈值，报告 F1/PR/AUC 曲线，默认 0 作保守基线。
- 真假对齐是否可靠？
  - 评估脚本对齐行列交集并统计样本数；输出 n_pairs 便于核查。
- 反卷积是否更好？
  - 原型阶段，展示趋势与可视化优势；需更多正则与超参搜索支撑系统性提升。

---

## 制作PPT的实际操作清单（不漏步骤）
1) 准备数据与环境
   - 确认安装 Scanpy、scikit-learn、PyTorch；记录 Python/CUDA 版本。
   - 放置 Xenium/Visium `.h5ad`、`spot_cluster_fraction_matrix.csv`、`x2v_mapping.csv`。
2) 生成标准 NPZ
   - 运行“页 3”命令，得到 `CellType_GP/data/spot_data_full.npz` 与真值输出。
3) 跑向量化与 LOFO
   - 运行“页 6/7”命令，得到 `pred_vectorized.csv` 和 `pred_lofo.csv`。
4) 跑反卷积原型（可选）
   - 运行“页 8”命令，得到 `deconv/` 产物与 `loss_curve.png`。
5) 评估与可视化
   - 运行“页 9”命令生成汇总与 per-feature 明细；
   - 挑选代表性 CT×Program 绘制散点与空间图（页 11）。
6) 组装结果页
   - 页 10：汇总表/柱状图；页 11：示例散点与热图；页 12：阈值与超参敏感性曲线；页 13：误差分析图。
7) 完成前检查
   - 指标是否与预期一致；对齐样本数 n_pairs 是否充分；命令可在 README 复现。
