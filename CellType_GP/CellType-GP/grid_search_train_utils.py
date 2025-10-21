#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
并行/多卡版：基于 train_model_2 的网格调参 + 导出 ctGP 宽表
----------------------------------------------------------------
- 对每个 (λ1, λ2, lr) 组合：
  1) 重新创建模型
  2) 调用 train_model_2 训练（早停+调度）
  3) 加载 best_model.pt
  4) 从 model.Y_tps 导出 ctGP 宽表 (S x T*P)，并用参数命名保存
- 多进程并行：自动按 GPU 数量轮询分配 CUDA_VISIBLE_DEVICES
- 只依赖你的 train_utils.py

用法示例：
  python grid_search_parallel_export.py \
      --npz ./DATA/spot_data_full.npz \
      --out ./grid_train_utils_parallel \
      --gpus 0,1 \
      --epochs 3000 --patience 200 \
      --l1 1e-5 5e-5 1e-4 5e-4 \
      --l2 5e-3 1e-2 2e-2 5e-2 \
      --lr 1e-3
"""

import os
import time
import json
from pathlib import Path
from itertools import product
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch

from train_utils import train_model_2


# ================ 需要你按项目实现的部分 ================
def create_model():
    """
    返回“全新初始化”的模型实例（每个组合必须是新实例）
    TODO: 将此处替换为你的模型构造，例如：
        from my_model import CTGPModel
        return CTGPModel(T=..., P=..., S=..., ...)
    """
    raise NotImplementedError("请实现 create_model() 返回一个新的模型实例。")


# ================ 工具函数 ================
def fmt(x: float) -> str:
    """把浮点数变成文件名友好的科学计数法（例如 1e-03）"""
    return f"{x:.2e}".replace("+", "").replace("0e", "e")


def make_tag(lr, l1, l2, ep):
    """组合标签"""
    return f"ep{ep}_lr{fmt(lr)}_l1{fmt(l1)}_l2{fmt(l2)}"


def load_npz(npz_path: Path):
    """
    读取 npz，返回：
      Y_obs (torch.Tensor)  —— 训练观测；注意维度需和你的 model.loss 匹配
      spot_names            —— (S,)
      celltype_names        —— (T,)
      program_names         —— (P,)
    你的 npz 应包含：
      - visium_score: (P, S)
      - spot_names, celltype_names, program_names
    """
    data = np.load(npz_path, allow_pickle=True)
    # 视你的模型约定决定是否需要转置
    Y_obs_np = data["visium_score"]  # (P, S)
    Y_obs = torch.tensor(Y_obs_np, dtype=torch.float32)

    spot_names = data["spot_names"]
    celltype_names = data["celltype_names"]
    program_names = data["program_names"]
    return Y_obs, spot_names, celltype_names, program_names


def export_ctgp_wide(model, spot_names, celltype_names, program_names, out_csv: Path):
    """
    从模型导出 Y_tps -> 宽表并保存
    约定：model 在训练后暴露 model.Y_tps (T, P, S)
    如果你的模型没有常驻属性，请改为 model.compute_Y_tps(...)
    """
    if not hasattr(model, "Y_tps"):
        raise RuntimeError("模型缺少 Y_tps 属性，请在模型中提供 Y_tps 或暴露 compute_Y_tps() 方法。")

    Y_tps = model.Y_tps.detach().cpu().numpy()  # (T, P, S)
    T, P, S = Y_tps.shape

    # 形状检查
    assert len(celltype_names) == T, f"T不匹配: {T} vs {len(celltype_names)}"
    assert len(program_names) == P, f"P不匹配: {P} vs {len(program_names)}"
    assert len(spot_names) == S, f"S不匹配: {S} vs {len(spot_names)}"

    # 转宽表 (S, T*P)
    Y_tps_flat = np.transpose(Y_tps, (2, 0, 1)).reshape(S, T * P)
    columns = [f"{ct}+{pg}" for ct in celltype_names for pg in program_names]
    df = pd.DataFrame(Y_tps_flat, index=spot_names, columns=columns)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=True)
    return df.shape


# ================ 单个组合的工作进程 ================
def run_one_combo(args):
    """
    子进程执行函数：
      - 设定 GPU 环境
      - 创建模型并训练
      - 加载 best_model 并导出 ctGP 宽表
      - 写 meta.json
    """
    (npz_path, base_outdir, dataset_tag, method_tag,
     lr, l1, l2, num_epochs, patience, tol, seed,
     gpu_id) = args

    # 进程内设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 进程内设置 GPU（如果有）
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # 载入数据
    Y_obs, spot_names, celltype_names, program_names = load_npz(Path(npz_path))
    Y_obs = Y_obs.to(device)

    # 生成目录与标签
    tag = make_tag(lr, l1, l2, num_epochs)
    run_dir = Path(base_outdir) / f"{dataset_tag}_{method_tag}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 创建模型
    model = create_model().to(device)

    t0 = time.time()
    try:
        history = train_model_2(
            model,
            Y_obs=Y_obs,
            num_epochs=num_epochs,
            lr=lr,
            lambda1=l1,
            lambda2=l2,
            early_stop=True,
            patience=patience,
            tol=tol,
            verbose=False,
            save_dir=str(run_dir),
        )
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "lambda1": l1, "lambda2": l2, "lr": lr,
            "num_epochs": num_epochs, "patience": patience, "tol": tol,
            "run_dir": str(run_dir)
        }

    # 加载最优权重
    best_pt = run_dir / "best_model.pt"
    if best_pt.exists():
        model.load_state_dict(torch.load(best_pt, map_location=device))
    model.eval()

    # 导出 ctGP 宽表
    ctgp_csv = run_dir / f"ctgp_{tag}.csv"
    shape = export_ctgp_wide(model, spot_names, celltype_names, program_names, ctgp_csv)

    # 写 meta
    meta = {
        "dataset": dataset_tag,
        "method": method_tag,
        "lambda1": l1, "lambda2": l2, "lr": lr,
        "num_epochs": num_epochs, "patience": patience, "tol": tol,
        "best_loss": float(history["best_loss"]),
        "best_epoch": int(history["best_epoch"]),
        "final_lr": float(history["lr"][-1]) if len(history["lr"]) else float(lr),
        "seconds": round(time.time() - t0, 2),
        "ctgp_csv": str(ctgp_csv),
        "ctgp_shape": list(shape),
        "run_dir": str(run_dir),
        "status": "ok",
    }
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


# ================ 并行调度入口 ================
def main():
    import argparse
    ap = argparse.ArgumentParser(description="并行/多卡网格调参 + 导出 ctGP 宽表（基于 train_model_2）")
    ap.add_argument("--npz", type=Path, required=True, help="包含 visium_score/spot_names/celltype_names/program_names 的 npz 路径")
    ap.add_argument("--out", type=Path, default=Path("./grid_train_utils_parallel"), help="输出根目录")
    ap.add_argument("--dataset", type=str, default="visium", help="数据标签（仅用于命名）")
    ap.add_argument("--method", type=str, default="train_utils_only", help="方法标签（仅用于命名）")

    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--l1", type=float, nargs="+", default=[1e-5, 5e-5, 1e-4, 5e-4], help="lambda1 候选列表")
    ap.add_argument("--l2", type=float, nargs="+", default=[5e-3, 1e-2, 2e-2, 5e-2], help="lambda2 候选列表")
    ap.add_argument("--lr", type=float, nargs="+", default=[1e-3], help="学习率候选列表")

    ap.add_argument("--gpus", type=str, default="", help="逗号分隔的 GPU id 列表，如 '0,1,2'；留空表示全用CPU")
    ap.add_argument("--workers", type=int, default=0, help="并行进程数；默认根据 GPU 数自动确定")
    args = ap.parse_args()

    # 设备列表与并行度
    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    if len(gpu_list) == 0:
        # 纯 CPU 并行
        devices = [None] * (args.workers if args.workers > 0 else max(1, mp.cpu_count() // 2))
    else:
        # 多卡：默认每张卡开一个进程；也可以用 --workers 控制更大并发
        n = args.workers if args.workers > 0 else len(gpu_list)
        devices = [gpu_list[i % len(gpu_list)] for i in range(n)]

    # 组合列表
    combos = list(product(args.l1, args.l2, args.lr))
    args.out.mkdir(parents=True, exist_ok=True)

    # 构造任务（把组合轮询分配给不同设备）
    tasks = []
    for idx, (l1, l2, lr) in enumerate(combos):
        gpu_id = devices[idx % len(devices)] if len(devices) > 0 else None
        tasks.append((
            str(args.npz),
            str(args.out),
            args.dataset,
            args.method,
            float(lr), float(l1), float(l2),
            int(args.epochs), int(args.patience), float(args.tol), int(args.seed),
            gpu_id
        ))

    print(f"🚀 总组合数: {len(combos)} | 并行度: {len(devices)} | 设备: {devices if devices else ['cpu']}")

    # 并行跑
    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=len(devices)) as pool:
        results = list(pool.map(run_one_combo, tasks))

    # 汇总 CSV
    df = pd.DataFrame(results)
    df.to_csv(args.out / "grid_summary.csv", index=False)
    print(f"\n📄 汇总已保存: {args.out / 'grid_summary.csv'}")

    # 打印最优
    ok = df[df["status"] == "ok"]
    if len(ok):
        best = ok.sort_values("best_loss", ascending=True).iloc[0]
        print("\n🏆 最优组合（按 best_loss 最小）：")
        print(
            f"  λ1={best['lambda1']}  λ2={best['lambda2']}  lr={best['lr']}\n"
            f"  best_loss={best['best_loss']:.6f} @ epoch {int(best['best_epoch'])}\n"
            f"  导出文件: {best['ctgp_csv']}\n  目录: {best['run_dir']}"
        )
    else:
        print("❗没有成功结果，请检查各 run_dir 下的 train_log.txt。")


if __name__ == "__main__":
    main()
