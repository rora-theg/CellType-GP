# Cell Type Specific Gene Program Deconvolution

## 目录结构
- `deconvolution/` : 代码模块
- `notebooks/` : Jupyter Notebook 示例
- `data/` : 示例数据（请自行准备或下载）
- `requirements.txt` : 依赖包

## 快速开始
1. 安装依赖：
```
pip install -r requirements.txt
```

2. 运行 Notebook：
```
jupyter notebook notebooks/01_run_pipeline.ipynb
```

3. 输入数据：
- `data/Y.npz` 文件，包含变量 Y, X, coords
  - `Y` : (P, S) 程序分数矩阵
  - `X` : (S, T) 细胞比例矩阵
  - `coords` : (S, 2) spot空间坐标

## 模块说明
- `model.py` : PyTorch模型
- `graph_utils.py` : 构建空间图拉普拉斯矩阵
- `train.py` : 训练流程
- `visualize.py` : 结果可视化

## 联系
如有问题欢迎交流。
