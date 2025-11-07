rm(list = ls())
gc()

setwd("/home/vs_theg/ST_program/CellType_GP/sc2spatial/DATA/folder")
load("./simulations//split.scRNAseq.forSimu.RData")

library(ggplot2)
library(Biobase)
library(SingleCellExperiment)
# 合并split1\split2
sc_merge <- Biobase::combine(split$eset.sub.split1, split$eset.sub.split2)
sc_merge

ct.varname <- "cellType" ## 细胞类型变量名
ct.select <- c("Astrocytes", "Neurons", "Oligos", "Vascular", "Immune", "Ependymal") ## 选择的细胞类型
sample.varname <- "sampleID" ## 样本ID变量名
# ---- 依赖 ----
library(Seurat)
# library(Biobase)
library(dplyr)
library(Matrix)

# ---- 从 ExpressionSet 构建 Seurat 对象 ----
expr_matrix <- Biobase::exprs(sc_merge) # gene x cell
meta_data <- Biobase::pData(sc_merge) # 细胞元信息（行名需与列名一致）
if (!all(colnames(expr_matrix) == rownames(meta_data))) {
    # 尽量对齐（保守处理）
    common <- intersect(colnames(expr_matrix), rownames(meta_data))
    expr_matrix <- expr_matrix[, common, drop = FALSE]
    meta_data <- meta_data[common, , drop = FALSE]
}

# 判断是否像原始计数（近似整数）
is_count_like <- function(m) {
    # 稀疏化判断 + 小误差浮点容忍
    s <- as.vector(m)
    if (length(s) > 2e6) s <- sample(s, 2e6) # 大矩阵抽样判断
    all(abs(s - round(s)) < 1e-8 & s >= 0)
}

if (is_count_like(expr_matrix)) {
    # 原始计数路径
    seu <- CreateSeuratObject(counts = expr_matrix, meta.data = meta_data, assay = "RNA")
    DefaultAssay(seu) <- "RNA"
    seu <- NormalizeData(seu) # LogNormalize 默认即可
    seu <- FindVariableFeatures(seu, selection.method = "vst", nfeatures = 3000)
    seu <- ScaleData(seu)
} else {
    # 已归一化表达路径（把矩阵放进 data 槽，不再 NormalizeData）
    seu <- CreateSeuratObject(counts = NULL, meta.data = meta_data, assay = "RNA")
    DefaultAssay(seu) <- "RNA"
    seu <- SetAssayData(seu, slot = "data", new.data = expr_matrix) # 直接作为 log/data
    seu <- FindVariableFeatures(seu, selection.method = "vst", nfeatures = 2000)
    seu <- ScaleData(seu) # 基于 data 槽做缩放/回归
}

# ---- 线性降维：PCA ----
set.seed(1234)
seu <- RunPCA(seu, npcs = 50, verbose = FALSE)

# 看 elbow 挑维度（手动一眼 20~40 常见）
png("ElbowPlot.png", width = 1200, height = 900, res = 150)
ElbowPlot(seu, ndims = 50) # dims取30
dev.off()


# ---- 图学习 & 非线性降维：UMAP ----
# 先用前 30 个主成分；如肘图更少/更多，可以改 dims
seu <- FindNeighbors(seu, dims = 1:30)
seu <- FindClusters(seu, resolution = 1) # 分辨率可微调：0.4（粗）~1.2（细）
seu <- RunTSNE(seu, dims = 1:20, n.neighbors = 30, min.dist = 0.6, 
verbose = FALSE, spread= 1.5, metric = "cosine",   # 对高维更稳
)

# 快速看图
png("tsne.png", width = 1200, height = 900, res = 150)
DimPlot(seu, reduction = "tsne", group.by ="cellType", label = TRUE)
dev.off()
# png("tsne.png", width = 1200, height = 900, res = 150)
# DimPlot(seu,reduction = "tsne",group.by = "seurat_clusters",label = T)
# dev.off()
# ---- 找每个细胞类型（或聚类）的 marker 基因 ----
# 优先用你自己的 celltype 注释；否则就用 seurat_clusters
group_col <- if ("cellType" %in% colnames(seu@meta.data)) "cellType" else "seurat_clusters"

# 注意：FindAllMarkers 默认用 DefaultAssay=RNA 的 data/scale
Idents(seu) <- group_col
markers <- FindAllMarkers(
    seu,
    only.pos = TRUE, # 只要上调基因
    min.pct = 0.25,
    logfc.threshold = 0.25,
    test.use = "wilcox" # 常见、稳健
)
# 每组取前10个做热图
top20 <- markers %>%
    group_by(cluster) %>%
    slice_max(order_by = avg_log2FC, n = 20)

# ---- 热图（每组Top10 marker）----
png("heatmap.png", width = 1600, height = 900, res = 150)
DoHeatmap(seu, features = unique(top20$gene), group.by = group_col) + NoLegend()
dev.off()
# ---- 细胞程序的一个“快捷入口”：模块打分（手工或自动法） ----
# 手工：你可以把某个功能基因集作为程序
# program_genes <- c("MKI67","TOP2A","PCNA","TYMS")  # 例如细胞增殖程序
# seu <- AddModuleScore(seu, features = list(program_genes), name = "Program_Prolif")
# FeaturePlot(seu, features = "Program_Prolif1")

# 或者，基于 marker 的共表达聚类后，把每个基因簇当“程序”再打分（需要你先分模块）
# 也可以后续上 cNMF/SCENIC 等自动提取程序

"""
由于生成模拟数据时为不同层设置了不同的主导细胞类型以及浓度参数，
这里采用每层的主导细胞类型的top10gene作为gene program set。
"""

## 第一层主导类型：神经元Neurons
Neurons_genes <-c("Sp9", "Igfbpl1", "Tubb3", "Gm27032", "Snhg11", 
                 "Dlx2", "Sncb", "Dlx1", "Nrsn1", "Slc32a1")

## 第2层主导类型：星形胶质细胞Astrocytes
Astrocytes_genes <- c("Slc7a10", "Aqp4", "Fgfr3", "Cldn10", "Slc6a11", 
                 "Ntsr2", "Slc25a18", "Acsbg1", "Mlc1", "Mgst1")

## 第3层主导类型：少突胶质细胞Oligos
Oligos_genes <- c("Tmem125","Opalin","Mog","Ermn","Nkx6-2",
                    "Mag","Enpp6","Fa2h","Cldn11","Hapln2")

## 在seu(total cells)中看看打分分布
seu <- AddModuleScore(seu, features = list(Neurons = Neurons_genes), name = "Neurons")
head(seu@meta.data)
png("Neurons.png", width =1200, height = 900, res = 150)

FeaturePlot(
  seu,
  features = "Neurons1",         # 模块分数列名
  reduction = "tsne",
  pt.size = 0.5
) + 
  ggtitle("Neurons Module Score") +
  theme_minimal()

dev.off()

seu <- AddModuleScore(seu, 
        features = list(Astrocytes = Astrocytes_genes), name = "Astrocytes")
png("Astrocytes.png", width =1200, height = 900, res = 150)
FeaturePlot(
  seu,
  features = "Astrocytes1",         # 模块分数列名
  reduction = "tsne",
  pt.size = 0.5
) + 
  ggtitle("Astrocytes Module Score") +
  theme_minimal()

dev.off()

seu <- AddModuleScore(seu, 
        features = list(Oligos =Oligos_genes), name = "Oligos")
png("Oligos.png", width =1200, height = 900, res = 150)
FeaturePlot(
  seu,
  features = "Oligos1",         # 模块分数列名
  reduction = "tsne",
  pt.size = 0.5
) + 
  ggtitle("Oligos Module Score") +
  theme_minimal()

dev.off()


# 1️⃣ Neurons marker
markers_neurons <- FindMarkers(
  seu, ident.1 = "Neurons", only.pos = TRUE,
  min.pct = 0.25, logfc.threshold = 0.25, test.use = "wilcox"
)
neurons_top30 <- head(markers_neurons[order(markers_neurons$avg_log2FC, decreasing = TRUE), ], 30)
neurons_top30_genes <- rownames(neurons_top30)

# 2️⃣ Astrocytes marker
markers_astro <- FindMarkers(
  seu, ident.1 = "Astrocytes", only.pos = TRUE,
  min.pct = 0.25, logfc.threshold = 0.25, test.use = "wilcox"
)
astro_top30 <- head(markers_astro[order(markers_astro$avg_log2FC, decreasing = TRUE), ], 30)
astro_top30_genes <- rownames(astro_top30)

# 3️⃣ Oligos marker
markers_oligo <- FindMarkers(
  seu, ident.1 = "Oligos", only.pos = TRUE,
  min.pct = 0.25, logfc.threshold = 0.25, test.use = "wilcox"
)
oligo_top30 <- head(markers_oligo[order(markers_oligo$avg_log2FC, decreasing = TRUE), ], 30)
oligo_top30_genes <- rownames(oligo_top30)

head(oligo_top30)

neurons_top30_genes
