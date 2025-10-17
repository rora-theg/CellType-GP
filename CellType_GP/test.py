#!/usr/bin/env python

## 读取数据
import pandas as pd 

train1500_result_wide=pd.read_csv("/home/vs_theg/ST_program/CellType_GP/DATA/train1500_result(wide).csv") #训练数据
truth_result_wide=pd.read_csv("/home/vs_theg/ST_program/CellType_GP/DATA/truth_result(wide).csv") #真实数据

## 其他反卷积的计算
import importlib
import sys
sys.path.append("/home/vs_theg/ST_program/CellType_GP/CellType-GP")
importlib.reload(sys.modules['celltype_gp_models'])  # 强制重新加载模块 
from celltype_gp_models import run_model


## 运行delta方法 (留一法)
df_pred = run_model(
    npz_path="/home/vs_theg/ST_program/CellType_GP/DATA/spot_data_full.npz",
    method="delta",
    save_path="/home/vs_theg/ST_program/CellType_GP/DATA/pred_result(delta_loo).csv"
)
pred_result_delta = pd.read_csv("/home/vs_theg/ST_program/CellType_GP/DATA/pred_result(delta_loo).csv", index_col=0)

