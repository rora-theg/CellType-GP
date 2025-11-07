# -*- coding: UTF-8 -*-
options(encoding = "UTF-8")

rm(list = ls())
gc()
####################################################################################################
## Package : CARD
## Version : 1.0.1
## Date    : 2022-5-7 09:10:08
## Title   : Main Simulation framework for CARD.
## Authors : Ying Ma
## Contacts: yingma@umich.edu
##           University of Michigan, Department of Biostatistics
####################################################################################################
## 该代码用于模拟空间转录组(spatial transcriptomics)数据集。
## 模拟的具体流程在主论文及补充材料中有详细描述。

## -------------------------------
## 1. 设置参数
## -------------------------------
iseed <- 3 ## 随机种子，保证可重复性
imix <- 0 ## 噪声混合参数（0表示无噪声）
ntotal <- 10 ## 每个空间位置包含的细胞总数

## -------------------------------
## 2. 加载单细胞RNA测序数据
## -------------------------------
## 将单细胞数据分为两部分：
## split1 用于模拟空间数据；
## split2 用于后续评估解卷积算法的性能。
## 数据过大，因此原始数据存放在 Google Drive 链接中。

setwd("/home/vs_theg/ST_program/CellType_GP/sc2spatial/DATA/folder")
load("./simulations//split.scRNAseq.forSimu.RData")
# eset.sub.split1 <- split$eset.sub.split1 ## 用于模拟空间数据
# eset.sub.split2 = split$eset.sub.split2 ## 用于下游分析
ct.varname <- "cellType" ## 细胞类型变量名
ct.select <- c("Astrocytes", "Neurons", "Oligos", "Vascular", "Immune") ## 选择的细胞类型 取出"Ependymal"
sample.varname <- "sampleID" ## 样本ID变量名

# 合并split1\split2
library(Biobase)
library(SingleCellExperiment)

sc_merge <- Biobase::combine(split$eset.sub.split1, split$eset.sub.split2)
dim(exprs(sc_merge))
# [1] 18263 20418
sc_merge
unique(sc_merge@phenoData$cellType)
table(sc_merge@phenoData$cellType)
table(pattern_gp_label)
## -------------------------------
## 3. 加载预定义的层标签与空间位置信息
## -------------------------------
load("./simulations/pattern_gp_label.RData")
load("./simulations/sim_MOB_location.RData")

## -------------------------------
## 4. 定义函数：用于生成模拟数据
## -------------------------------

#### 函数1：从Dirichlet分布生成随机数（用于生成细胞类型比例）
generateMultiN <- function(pattern_gp_label, ipt, ntotal, mix, ct.select) {
  library(MCMCpack)
  message(paste0("Generating cell type proportions for pattern", ipt))
  nobs <- sum(pattern_gp_label == ipt) ## 当前层中spot数量
  sample <- matrix(0, nrow = nobs, ncol = length(ct.select))
  colnames(sample) <- ct.select
  sampleNames <- names(pattern_gp_label)[pattern_gp_label == ipt]
  prop <- matrix(0, nrow = nobs, ncol = length(ct.select))
  colnames(prop) <- ct.select

  ## 随机确定当前层的细胞类型数量（主细胞类型 + 混合类型）
  numCT <- sample(1:length(ct.select), 1)

  ## 为不同层设置主导细胞类型(main)及浓度参数
  if (ipt == 1) {
    main <- "Neurons" ### 第1层主导类型：神经元
    concen <- rep(1, numCT)
  } else if (ipt == 2) {
    main <- "Astrocytes" ### 第2层主导类型：星形胶质细胞
    concen <- rep(1, numCT)
  } else if (ipt == 3) {
    main <- "Oligos" ### 第3层主导类型：少突胶质细胞
    concen <- rep(1, numCT)
  }

  ## 从Dirichlet分布生成比例
  propSample <- rdirichlet(nrow(prop), concen)

  ## 随机选择次要细胞类型
  ct.select.sub.sample <- sample(ct.select[ct.select != main], numCT - 1)
  ct.select.sub <- c(main, ct.select.sub.sample)

  ## 按mix比例确定噪声点与正常点
  Fix_Dirichlet <- sample(1:nrow(sample), round(nobs * mix[1]))
  mix_Dirichlet <- c(1:nrow(sample))[!(c(1:nrow(sample)) %in% Fix_Dirichlet)]

  ## 对噪声点重新采样Dirichlet分布
  if (length(mix_Dirichlet) > 0) {
    propSample[mix_Dirichlet, ] <- rdirichlet(length(mix_Dirichlet), rep(1, numCT))
  }

  print(ct.select.sub)

  ## 对非噪声点，将主细胞类型的比例设为最大
  if (length(Fix_Dirichlet) > 0) {
    propSample[Fix_Dirichlet, ] <- t(sapply(Fix_Dirichlet, function(i) {
      propSample[i, ][order(propSample[i, ], decreasing = T)]
    }))
  }

  ## 将比例赋值到总矩阵中
  colnames(propSample) <- ct.select.sub
  prop[, ct.select.sub] <- propSample
  sample <- round(ntotal * prop, digits = 0)

  ##### 确保每个spot至少有一个细胞
  index <- which(rowSums(sample) == 0)
  if (length(index) > 0) {
    sample[index, ] <- t(sapply(index, function(i) {
      rmultinom(1, ntotal, prob = prop[i, ])
    }))
  }

  return(list(sample = sample))
}

#### 函数2：根据给定比例生成空间数据矩阵，并记录单细胞到空间点映射
generateSpatial_norep_fixedProp <- function(
    seed,
    sc_merge,
    ct.varname,
    sample.varname,
    ct.select,
    sample.withRep = FALSE,
    pattern_gp_label,
    ntotal,
    mix1,
    mix2,
    mix3,
    allow_reuse = TRUE) {
  # 使用 split1 抽取单细胞数据生成伪空间数据
  phenoData <- sc_merge@phenoData@data
  k <- length(unique(ct.select))
  message(paste("Using", k, "cell types to generate pseudo spatial dataset"))

  # 初始化矩阵：每个空间位置的细胞类型组成
  Sample_random <- matrix(0, nrow = length(pattern_gp_label), ncol = length(ct.select))
  rownames(Sample_random) <- names(pattern_gp_label)
  colnames(Sample_random) <- ct.select

  # 获取细胞类型信息
  ct.id <- droplevels(as.factor(sc_merge@phenoData@data[, ct.varname]))
  library(Hmisc)

  # 分别生成三层的细胞类型分布
  pattern1 <- generateMultiN(pattern_gp_label, 1, ntotal, mix1, ct.select)
  pattern2 <- generateMultiN(pattern_gp_label, 2, ntotal, mix2, ct.select)
  pattern3 <- generateMultiN(pattern_gp_label, 3, ntotal, mix3, ct.select)

  # 将三层的分布合并
  Sample_random[pattern_gp_label == 1, ] <- pattern1$sample
  Sample_random[pattern_gp_label == 2, ] <- pattern2$sample
  Sample_random[pattern_gp_label == 3, ] <- pattern3$sample

  message(paste0("Generating pseudo spatial dataset for ", length(unique(pattern_gp_label)), " patterns"))

  # 设置随机种子
  set.seed(seed)
  temp.exprs <- exprs(sc_merge) # 表达矩阵 (基因 x 细胞)
  temp.nct <- Sample_random # 每个 spot 的细胞数量矩阵
  true.p <- sweep(temp.nct, 1, rowSums(temp.nct), "/") # 转换为比例
  all_cell_ids <- colnames(temp.exprs)

  if (allow_reuse) {
    message("allow_reuse=TRUE: 不同 spot 可共享同一单细胞（仅在细胞不足时会放回抽样）。")

    temp_result_list <- pbmclapply(
      X = seq_len(nrow(temp.nct)),
      FUN = function(isample) {
        spot_name <- rownames(temp.nct)[isample]

        temp_sample_list <- lapply(ct.select, function(ict) {
          idx_ct <- which(ct.id %in% ict)
          temp.vec <- temp.exprs[, idx_ct, drop = FALSE]
          n_pick <- temp.nct[isample, ict]
          sample_flag <- sample.withRep
          if (n_pick > ncol(temp.vec)) {
            sample_flag <- TRUE
          }

          if (n_pick > 0 && ncol(temp.vec) > 0) {
            temp.id <- sample(seq_len(ncol(temp.vec)), n_pick, replace = sample_flag)
            picked_cell_ids <- all_cell_ids[idx_ct[temp.id]]
            expr_block <- temp.vec[, temp.id, drop = FALSE]
            if (NCOL(expr_block) > 1) {
              expr_block <- rowSums(expr_block)
            }
            list(
              expr = as.matrix(expr_block),
              mapping = data.frame(
                cell_id = picked_cell_ids,
                spot_id = rep(spot_name, length(picked_cell_ids)),
                celltype = rep(ict, length(picked_cell_ids)),
                stringsAsFactors = FALSE
              )
            )
          } else {
            list(
              expr = matrix(0, nrow = nrow(temp.exprs), ncol = 1),
              mapping = data.frame(
                cell_id = character(),
                spot_id = character(),
                celltype = character(),
                stringsAsFactors = FALSE
              )
            )
          }
        })

        expr_mat <- do.call(cbind, lapply(temp_sample_list, function(x) x$expr))
        mapping_df <- do.call(rbind, lapply(temp_sample_list, function(x) x$mapping))

        list(expr_sum = rowSums(expr_mat), mapping = mapping_df)
      },
      mc.cores = 70,
      mc.set.seed = FALSE
    )

    temp.pseudo <- do.call("cbind", lapply(temp_result_list, function(x) x$expr_sum))
    colnames(temp.pseudo) <- rownames(Sample_random)
    rownames(temp.pseudo) <- rownames(temp.exprs)

    mapping_df <- do.call(rbind, lapply(temp_result_list, function(x) x$mapping))
  } else {
    message("allow_reuse=FALSE: 不同 spot 不共享单细胞，按类型逐一消耗可用细胞池。")

    available_cells <- lapply(ct.select, function(ict) which(ct.id %in% ict))
    names(available_cells) <- ct.select

    pseudo_list <- vector("list", nrow(temp.nct))
    mapping_list <- vector("list", nrow(temp.nct))

    for (isample in seq_len(nrow(temp.nct))) {
      spot_name <- rownames(temp.nct)[isample]
      spot_expr <- numeric(nrow(temp.exprs))
      spot_map <- data.frame(
        cell_id = character(),
        spot_id = character(),
        celltype = character(),
        stringsAsFactors = FALSE
      )

      for (ict in ct.select) {
        n_pick <- temp.nct[isample, ict]
        if (n_pick == 0) {
          next
        }

        pool_idx <- available_cells[[ict]]
        if (length(pool_idx) < n_pick) {
          stop(sprintf(
            "细胞类型 %s 可用单细胞不足：需要 %d，剩余 %d。请减少 ntotal 或允许重复抽样。",
            ict, n_pick, length(pool_idx)
          ))
        }

        chosen_idx <- sample(pool_idx, n_pick, replace = FALSE)
        available_cells[[ict]] <- setdiff(pool_idx, chosen_idx)
        picked_cell_ids <- all_cell_ids[chosen_idx]

        expr_block <- temp.exprs[, chosen_idx, drop = FALSE]
        if (NCOL(expr_block) > 1) {
          spot_expr <- spot_expr + rowSums(expr_block)
        } else {
          spot_expr <- spot_expr + expr_block[, 1]
        }

        spot_map <- rbind(
          spot_map,
          data.frame(
            cell_id = picked_cell_ids,
            spot_id = rep(spot_name, length(picked_cell_ids)),
            celltype = rep(ict, length(picked_cell_ids)),
            stringsAsFactors = FALSE
          )
        )
      }

      pseudo_list[[isample]] <- spot_expr
      mapping_list[[isample]] <- spot_map
    }

    temp.pseudo <- do.call("cbind", pseudo_list)
    colnames(temp.pseudo) <- rownames(Sample_random)
    rownames(temp.pseudo) <- rownames(temp.exprs)

    mapping_df <- do.call(rbind, mapping_list)
  }

  list(
    pseudo.data = temp.pseudo,
    true.p = true.p,
    ntotal = ntotal,
    Sample_random = Sample_random,
    mapping = mapping_df
  )
}

table(table(sc_merge@phenoData@data[, ct.varname]))

## -------------------------------
## 5. 模拟数据执行部分
## -------------------------------

library(pbmcapply)
library(SingleCellExperiment)

## 设置噪声比例
## imix = 0,1,2,3 对应 0%,20%,40%,60%
mix1 <- mix2 <- mix3 <- c(1 - (0.2 * imix), 0.2 * imix)
set.seed(iseed)

## 生成模拟数据
spatial.pseudo <- generateSpatial_norep_fixedProp(
  seed = iseed,
  sc_merge = sc_merge,
  ct.varname = ct.varname,
  sample.varname = sample.varname,
  ct.select = ct.select,
  sample.withRep = FALSE,
  pattern_gp_label = pattern_gp_label,
  ntotal = ntotal,
  mix1 = mix1,
  mix2 = mix2,
  mix3 = mix3,
  allow_reuse = FALSE
)


## -------------------------------
## 6. 检查模拟结果
## -------------------------------

## 平均每个spot的细胞数
print(round(mean(rowSums(spatial.pseudo$Sample_random), 0)))
# 10

## 模拟数据的维度（基因 × 空间点）
print(dim(spatial.pseudo$pseudo.data))
# [1] 18263   260

## 每种细胞类型的平均比例
print(colMeans(spatial.pseudo$true.p))
# Astrocytes    Neurons     Oligos   Vascular     Immune  Ependymal
# 0.07065657 0.25750971 0.38714744 0.05901515 0.10253594 0.12313520

## 所有用于Figure 2的模拟数据可在以下地址找到：
## https://drive.google.com/drive/folders/1wRPxn1YI7f1oUw8eC42htXMjTUqyIT1g
## 然后可以继续运行 CARD

## --------------------------------------
## 7. 映射与配对单细胞数据保存
## --------------------------------------
sum(duplicated(mapping$cell_id))

# 直接使用函数返回的映射（与伪空间表达严格一致）
mapping <- spatial.pseudo$mapping
outdir <- "/home/vs_theg/ST_program/CellType_GP/sc2spatial/DATA/sim_ctgp/"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
write.csv(mapping, file = file.path(outdir, "cell_to_spot_mapping.csv"), row.names = FALSE)
message("✅ Saved raw mapping: ", nrow(mapping), " rows to ", outdir)
# ✅ Saved raw mapping: 2559 rows to /home/vs_theg/ST_program/CellType_GP/sc2spatial/DATA/sim_ctgp/

# 基于映射抽取配对的单细胞，生成一个新的数据集（与空间数据一一对应）
paired_cell_ids <- unique(mapping$cell_id)
paired_idx <- match(paired_cell_ids, colnames(exprs(sc_merge)))
paired_idx <- paired_idx[!is.na(paired_idx)]
paired_sc <- sc_merge[, paired_idx]
save(paired_sc, file = file.path(outdir, "sc_2559.RData"))
message("✅ Saved paired single-cell object with ", ncol(exprs(paired_sc)), " cells")

# 同步导出单细胞表达矩阵（稀疏矩阵）与元数据，便于后续分析
paired_expr <- exprs(paired_sc)
saveRDS(paired_expr, file = file.path(outdir, "sc_2559.rds")) # rds元数据
paired_meta <- Biobase::pData(paired_sc)
paired_meta$cell_id <- rownames(paired_meta)
write.csv(paired_meta, file = file.path(outdir, "paired_singlecell_metadata.csv"), row.names = FALSE) # 单细胞表达矩阵
message("✅ Exported paired single-cell expression (RDS) 与 metadata (CSV)")

# spot - cell 映射汇总表，便于核查每个 spot 对应的细胞列表
library(dplyr)
mapping_summary <- mapping %>%
  group_by(spot_id) %>%
  summarise(cell_ids = paste(cell_id, collapse = ";"), .groups = "drop")
write.csv(mapping_summary, file = file.path(outdir, "spot_to_cells.csv"), row.names = FALSE) # spot2cells_mapping
message("✅ Exported spot_to_cells summary for quick lookup.")

## -------------------------------
## 8. 预处理模拟数据并保存
## -------------------------------
head(location)
coords <- location
coords$id <- paste0(coords$x, "x", coords$y)
coords$spot_id <- sprintf("spot%03d", seq_len(nrow(coords))) # spot001, spot002, ...
# 创建 id（x,y） 到 spot_id 的映射
id_map <- setNames(coords$spot_id, coords$id)

# 替换表达矩阵的列名
colnames(spatial.pseudo$pseudo.data) <- id_map[colnames(spatial.pseudo$pseudo.data)]
# 替换 pseudo 数据框的行名
rownames(spatial.pseudo$Sample_random) <- id_map[rownames(spatial.pseudo$Sample_random)]
rownames(spatial.pseudo$true.p) <- id_map[rownames(spatial.pseudo$true.p)]

write.csv(spatial.pseudo$pseudo.data,
  file = "/home/vs_theg/ST_program/CellType_GP/sc2spatial/DATA/sim_ctgp/pseudo_data.csv"
)
write.csv(spatial.pseudo$true.p,
  file = "/home/vs_theg/ST_program/CellType_GP/sc2spatial/DATA/sim_ctgp/true_p.csv"
)
write.csv(spatial.pseudo$Sample_random,
  file = "/home/vs_theg/ST_program/CellType_GP/sc2spatial/DATA/sim_ctgp/sample_random.csv"
)


mapping2 <- mapping
# ---- 替换 mapping 表的 spot_id 列（与上方重命名保持一致） ----
mapping2$spot_id <- id_map[mapping2$spot_id]
write.csv(mapping2, file.path(outdir, "cell_to_spot_mapping.csv"), row.names = FALSE)
message("✅ Saved renamed mapping with spot ids: ", nrow(mapping2))
