"""
计算 Xenium 数据的真实基因程序得分（truth score）
函数化版本：可清洗、分组、透视、保存
作者：theg（本次改动：去交互化入口、使用清洗后的 DataFrame、可配置分组列名）
"""

import scanpy as sc
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler


# ========== 函数定义区 ==========

def clean_obs_data(adata, drop_columns: list[str] = None):
    """
    清洗 adata.obs：
    - 输出所有列名为一行字符串（可直接复制粘贴）
    - 用户输入要删除的列（直接粘贴即可）
    - 自动删除并输出清洗结果
    """

    df = adata.obs.copy()

    # 1️⃣ 输出列名，方便复制粘贴
    col_list = ", ".join(df.columns)
    print("\n🧾 当前 adata.obs 列如下（可直接复制粘贴）：\n")
    print(col_list)
    print("\n💡 提示：你可以复制上面这一行，然后粘贴要删除的列（或部分列）")

    # 2️⃣ 去交互化（如未提供 drop_columns，则不删除；保留交互作为 fallback）
    if drop_columns is None:
        try:
            user_input = input("\n请输入要删除的列（多个用逗号分隔，直接回车则跳过删除）:\n> ").strip()
            if user_input:
                drop_columns = [c.strip() for c in user_input.split(",") if c.strip()]
            else:
                drop_columns = []
        except Exception:
            # 非交互环境：忽略删除
            drop_columns = []

    # 3️⃣ 执行删除
    if drop_columns:
        df.drop(columns=drop_columns, inplace=True, errors='ignore')
        print(f"\n🧹 已删除 {len(drop_columns)} 列：{', '.join(drop_columns)}")
    else:
        print("\n✅ 未删除任何列")

    # 4️⃣ 额外清理
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    print(f"✨ 清洗完成：保留 {df.shape[1]} 列。\n")
    return df


def compute_group_means(df: pd.DataFrame, spot_col: str, celltype_col: str,
                        suffix: str, ## 目标列后缀，自定义
                        normalize_within_spot:bool = True
                        ):
    """
    按 Visium barcode(spot_col) + broad_annotation(celltype_col) 分组，计算各 _score 列的平均值
    返回 group 平均与计数。
    参数：
        df : pd.DataFrame
            包含 spot 和 celltype 的数据框。
        spot_col : str
            斑点（barcode）列名。
        celltype_col : str
            细胞类型列名。
        suffix : str, 默认 '_score'
            目标列的后缀（如 '_score'、'_norm'），用于筛选计算列。
    
    返回：
        pd.DataFrame : 合并的平均值 + 计数 DataFrame。
    """

    # 筛选以 suffix 结尾的列
    # 筛选目标列
    score_cols = [col for col in df.columns if col.endswith(suffix)]
    if len(score_cols) == 0:
        raise ValueError(f"❌ 未找到以 '{suffix}' 结尾的列。")

    print(f"🧩 检测到 {len(score_cols)} 个目标列（后缀 '{suffix}'）")

    grouped_means = (
        df.groupby([spot_col, celltype_col], observed=False)[score_cols]
        .mean()
        .reset_index()
    )
    grouped_counts = (
        df.groupby([spot_col, celltype_col], observed=False)
        .size()
        .reset_index(name='cell_count')
    )
    truth_result = pd.merge(grouped_means, grouped_counts, on=[spot_col, celltype_col])

    # ✅ Spot 内归一化
    if normalize_within_spot:
        print("⚙️ 正在对每个 spot 内 gene program 均值进行 MinMax 归一化 ...")
        def normalize_group(x):
            scaler = MinMaxScaler()
            x[score_cols] = scaler.fit_transform(x[score_cols])
            return x
        truth_result = truth_result.groupby(spot_col, group_keys=False, observed=False).apply(normalize_group)
        print("✅ 归一化完成（按每个 spot 进行 MinMax 缩放）")

    print(f"✅ 分组平均完成: {truth_result.shape[0]} 行")
    return truth_result

                       
def pivot_truth_scores(truth_result: pd.DataFrame, spot_col: str, celltype_col: str,
                       suffix:str ):
    """将 truth_result 转为宽表 (spot × celltype+program)"""
    program_cols = [c for c in truth_result.columns if c.endswith(suffix)]

    truth_wide = truth_result.pivot_table(index=spot_col, columns=celltype_col, values=program_cols)

    # 展开多级列名
    truth_wide.columns = [f"{ctype}+{pg}" for pg, ctype in truth_wide.columns]
    truth_wide = truth_wide.reset_index().rename(columns={spot_col: "spot"})

    print(f"✅ 宽表完成: {truth_wide.shape[0]} × {truth_wide.shape[1]}")
    return truth_wide


def save_truth_outputs(df_clean: pd.DataFrame,
                       truth_result: pd.DataFrame,
                       truth_wide: pd.DataFrame,
                       output_dir: str):
    """保存所有结果文件"""
    os.makedirs(output_dir, exist_ok=True)
    path_score = os.path.join(output_dir, "truth_score.csv")
    path_result = os.path.join(output_dir, "truth_result_grouped.csv")
    path_wide = os.path.join(output_dir, "truth_result(wide).csv")

    df_clean.to_csv(path_score, index=False)
    truth_result.to_csv(path_result, index=False)
    truth_wide.to_csv(path_wide, index=False)

    print("💾 保存结果：")
    print(f"  ├─ 细胞级 truth_score：{path_score}")
    print(f"  ├─ 分组均值 truth_result：{path_result}")
    print(f"  └─ 宽表 truth_result(wide)：{path_wide}")


# ========== 主函数入口 ==========

def compute_truth_score(adata, output_dir: str = "./truth_output",
                       spot_col: str = 'transcript_level_visium_barcode',
                       celltype_col: str = 'broad_annotation',
                       drop_columns: list[str] = None,
                       suffix: str = '_score'):
    """
    从已加载的 Xenium AnnData 对象计算真实基因程序得分。
    参数:
        adata : 已加载的 AnnData 对象
        output_dir : 输出目录（默认 ./truth_output）
    """
    print(f"🚀 开始计算 Xenium truth score, 输出路径: {output_dir}")

    # Step 1️⃣ 清洗 obs（非交互环境可传入 drop_columns=None 以保持原样）
    df_clean = clean_obs_data(adata, drop_columns=drop_columns)

    # Step 2️⃣ 计算分组均值（使用清洗后的 df_clean）
    truth_result = compute_group_means(df_clean, spot_col=spot_col, celltype_col=celltype_col, suffix=suffix)

    # Step 3️⃣ 转宽表
    truth_wide = pivot_truth_scores(truth_result, spot_col=spot_col, celltype_col=celltype_col, suffix=suffix)

    # Step 4️⃣ 保存结果
    save_truth_outputs(df_clean, truth_result, truth_wide, output_dir)

    print("\n🎉 Xenium truth score 计算完成！")
    return truth_wide
