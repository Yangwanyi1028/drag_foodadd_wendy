#!/usr/bin/env Rscript
# 绘制Top Ranking特征的箱线图
# 特别关注Sac_S和Suc_S在Discovery和Validation两个cohort中的分布对比
# 参考: /Users/yangkeyi/Downloads/predict_relapse/predict_flare_in_1y_CD/05model_improved/plot_selected_features.R

rm(list = ls())

# 加载必要的包
suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(RColorBrewer)
})

# 设置工作目录
setwd("/Users/yangkeyi/Downloads/diagnostic_model_wendy_0126pm")

# 配置参数
N_TOP_FEATURES <- 15  # 显示top N特征
OUTPUT_DIR <- "results/feature_boxplots"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# 颜色定义
GROUP0_COLOR <- "#2E86AB"  # 蓝色 - Group 0 (健康/对照)
GROUP1_COLOR <- "#A23B72"  # 紫色 - Group 1 (病人)
COHORT_COLORS <- c("Discovery" = "#8B5CF6", "Validation" = "#F18F01")

# ============================================================================
# 读取数据
# ============================================================================
message("读取数据...")

# 读取特征排名
feature_ranking <- read.csv("v4_feature_ranking.csv", stringsAsFactors = FALSE)
feature_ranking <- feature_ranking[order(feature_ranking$Avg_rank), ]
top_features <- head(feature_ranking$X, N_TOP_FEATURES)

message(sprintf("\n选定的 Top %d 个特征:", length(top_features)))
for (i in seq_along(top_features)) {
  message(sprintf("  %d. %s (Avg_rank=%.2f)", i, top_features[i], 
                  feature_ranking[feature_ranking$X == top_features[i], "Avg_rank"]))
}

# 特别标记Sac_S和Suc_S
special_features <- c("Sac_S", "Suc_S")
message(sprintf("\n特别关注的特征: %s", paste(special_features, collapse = ", ")))

# 读取Discovery数据
disc_df <- read.csv("data/Discovery_P80_sweetners_CD_individual_activity(1).csv", 
                    na.strings = c('.', '', ' '), stringsAsFactors = FALSE)
disc_df$Group <- as.numeric(disc_df$Group)
disc_df <- disc_df %>% filter(!is.na(Group))
disc_df$Group <- as.integer(disc_df$Group)
disc_df$Cohort <- "Discovery"

# 读取Validation数据
val_df <- read.csv("data/Validation_P80_sweetners_CD_individual_activity_Re.csv", 
                   na.strings = c('.', '', ' '), stringsAsFactors = FALSE)
val_df$Group <- as.numeric(val_df$Group)
val_df <- val_df %>% filter(!is.na(Group))
val_df$Group <- as.integer(val_df$Group)
val_df$Cohort <- "Validation"

message(sprintf("\n数据加载完成:"))
message(sprintf("  Discovery: %d samples (Group 0: %d, Group 1: %d)", 
                nrow(disc_df), 
                sum(disc_df$Group == 0), 
                sum(disc_df$Group == 1)))
message(sprintf("  Validation: %d samples (Group 0: %d, Group 1: %d)", 
                nrow(val_df), 
                sum(val_df$Group == 0), 
                sum(val_df$Group == 1)))

# ============================================================================
# 准备绘图数据
# ============================================================================
message("\n准备绘图数据...")

plot_data_list <- list()

for (feat in top_features) {
  # 检查特征是否存在于两个数据集中
  if (!(feat %in% colnames(disc_df)) || !(feat %in% colnames(val_df))) {
    message(sprintf("  跳过 %s (不存在于数据中)", feat))
    next
  }
  
  # Discovery数据
  disc_feat <- disc_df %>%
    select(sampleID, Group, Cohort, all_of(feat)) %>%
    rename(value = all_of(feat)) %>%
    mutate(value = as.numeric(value))
  
  # Validation数据
  val_feat <- val_df %>%
    select(sampleID, Group, Cohort, all_of(feat)) %>%
    rename(value = all_of(feat)) %>%
    mutate(value = as.numeric(value))
  
  # 合并
  feat_data <- bind_rows(disc_feat, val_feat) %>%
    filter(!is.na(value)) %>%
    mutate(
      feature = feat,
      Group_label = ifelse(Group == 0, "Group 0", "Group 1"),
      Group_label = factor(Group_label, levels = c("Group 0", "Group 1")),
      Cohort = factor(Cohort, levels = c("Discovery", "Validation")),
      is_special = feat %in% special_features
    )
  
  # 对数转换（处理0值）
  feat_data$log_value <- log10(ifelse(feat_data$value == 0, 1e-6, feat_data$value))
  
  # 添加统计信息（从feature_ranking）
  if (feat %in% feature_ranking$X) {
    rank_row <- feature_ranking[feature_ranking$X == feat, ]
    feat_data$MW_score <- rank_row$MW_score
    feat_data$Effect <- rank_row$Effect
    feat_data$Avg_rank <- rank_row$Avg_rank
  }
  
  plot_data_list[[feat]] <- feat_data
}

plot_data_all <- bind_rows(plot_data_list)

# ============================================================================
# 绘制箱线图 - 单个特征（Discovery vs Validation对比）
# ============================================================================
message("\n绘制箱线图...")

boxplot_dir <- file.path(OUTPUT_DIR, "individual_features")
dir.create(boxplot_dir, showWarnings = FALSE)

for (feat in names(plot_data_list)) {
  feat_data <- plot_data_list[[feat]]
  is_special <- feat %in% special_features
  
  # 计算统计信息
  disc_g0 <- feat_data %>% filter(Cohort == "Discovery", Group == 0) %>% pull(value)
  disc_g1 <- feat_data %>% filter(Cohort == "Discovery", Group == 1) %>% pull(value)
  val_g0 <- feat_data %>% filter(Cohort == "Validation", Group == 0) %>% pull(value)
  val_g1 <- feat_data %>% filter(Cohort == "Validation", Group == 1) %>% pull(value)
  
  # 计算均值和中位数
  disc_g0_mean <- mean(disc_g0, na.rm = TRUE)
  disc_g1_mean <- mean(disc_g1, na.rm = TRUE)
  val_g0_mean <- mean(val_g0, na.rm = TRUE)
  val_g1_mean <- mean(val_g1, na.rm = TRUE)
  
  disc_direction <- ifelse(disc_g1_mean > disc_g0_mean, "G1 > G0", "G0 > G1")
  val_direction <- ifelse(val_g1_mean > val_g0_mean, "G1 > G0", "G0 > G1")
  direction_consistent <- disc_direction == val_direction
  
  # 样本量
  n_disc_g0 <- length(disc_g0)
  n_disc_g1 <- length(disc_g1)
  n_val_g0 <- length(val_g0)
  n_val_g1 <- length(val_g1)
  
  # 获取排名信息
  avg_rank <- feat_data$Avg_rank[1]
  mw_score <- feat_data$MW_score[1]
  effect <- feat_data$Effect[1]
  
  # 创建统计信息标签
  stats_label <- sprintf(
    "Avg Rank: %.2f\nMW Score: %.2f\nEffect: %.3f\n\nDiscovery: %s\nValidation: %s",
    avg_rank, mw_score, effect, disc_direction, val_direction
  )
  
  if (!direction_consistent) {
    stats_label <- paste0(stats_label, "\n⚠️ DIRECTION REVERSED!")
  }
  
  # 绘制箱线图
  p <- ggplot(feat_data, aes(x = Cohort, y = log_value, fill = Group_label)) +
    # 箱线图
    geom_boxplot(
      position = position_dodge(width = 0.8),
      outlier.shape = NA,
      alpha = 0.7,
      width = 0.6,
      color = "black",
      linewidth = 0.5
    ) +
    # 散点（jitter）
    geom_jitter(
      aes(color = Group_label),
      position = position_jitterdodge(dodge.width = 0.8, jitter.width = 0.2),
      size = 2,
      alpha = 0.6
    ) +
    # 颜色设置
    scale_fill_manual(
      values = c("Group 0" = GROUP0_COLOR, "Group 1" = GROUP1_COLOR),
      name = "Group"
    ) +
    scale_color_manual(
      values = c("Group 0" = GROUP0_COLOR, "Group 1" = GROUP1_COLOR),
      guide = "none"
    ) +
    # 标签
    labs(
      title = feat,
      subtitle = ifelse(is_special, "⚠️ SPECIAL ATTENTION", ""),
      x = "Cohort",
      y = "Value (log10)",
      fill = "Group"
    ) +
    # 主题
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(face = "bold", size = 12, hjust = 0.5, color = "red"),
      axis.text = element_text(color = "black", size = 12),
      axis.title = element_text(size = 13),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", linewidth = 1),
      legend.position = "right",
      plot.background = element_rect(fill = "white", color = NA),
      plot.margin = margin(10, 15, 10, 15)
    )
  
  # 添加统计信息（右上角）
  p <- p + annotate(
    "text",
    x = Inf, y = Inf,
    label = stats_label,
    hjust = 1.1, vjust = 1.1,
    size = 3,
    color = ifelse(direction_consistent, "black", "red"),
    fontface = ifelse(direction_consistent, "plain", "bold")
  )
  
  # 保存图片
  filename <- gsub("[^A-Za-z0-9_]", "_", feat)
  output_file <- file.path(boxplot_dir, paste0("boxplot_", filename, ".png"))
  
  ggsave(
    output_file,
    p,
    width = 8,
    height = 6,
    dpi = 300,
    bg = "white"
  )
  
  if (is_special) {
    message(sprintf("  ✓ [SPECIAL] %s: Discovery %s, Validation %s", 
                    feat, disc_direction, val_direction))
  }
}

message(sprintf("\n✓ 箱线图已保存到: %s", boxplot_dir))

# ============================================================================
# 绘制组合图 - Sac_S和Suc_S的详细对比
# ============================================================================
message("\n绘制Sac_S和Suc_S的详细对比图...")

special_data <- plot_data_all %>% filter(is_special)

if (nrow(special_data) > 0) {
  p_special <- ggplot(special_data, aes(x = Cohort, y = log_value, fill = Group_label)) +
    geom_boxplot(
      position = position_dodge(width = 0.8),
      outlier.shape = NA,
      alpha = 0.7,
      width = 0.6,
      color = "black",
      linewidth = 0.5
    ) +
    geom_jitter(
      aes(color = Group_label),
      position = position_jitterdodge(dodge.width = 0.8, jitter.width = 0.2),
      size = 2.5,
      alpha = 0.6
    ) +
    scale_fill_manual(
      values = c("Group 0" = GROUP0_COLOR, "Group 1" = GROUP1_COLOR),
      name = "Group"
    ) +
    scale_color_manual(
      values = c("Group 0" = GROUP0_COLOR, "Group 1" = GROUP1_COLOR),
      guide = "none"
    ) +
    facet_wrap(~ feature, scales = "free_y", ncol = 2) +
    labs(
      title = "Sac_S and Suc_S: Discovery vs Validation Comparison",
      subtitle = "⚠️ Check for direction consistency between cohorts",
      x = "Cohort",
      y = "Value (log10)",
      fill = "Group"
    ) +
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(face = "bold", size = 12, hjust = 0.5, color = "red"),
      axis.text = element_text(color = "black", size = 11),
      axis.title = element_text(size = 13),
      strip.text = element_text(face = "bold", size = 12),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", linewidth = 1),
      legend.position = "right"
    )
  
  ggsave(
    file.path(OUTPUT_DIR, "boxplot_Sac_S_Suc_S_comparison.png"),
    p_special,
    width = 12,
    height = 6,
    dpi = 300,
    bg = "white"
  )
  
  message("✓ 特殊特征对比图已保存")
}

# ============================================================================
# 生成统计摘要表
# ============================================================================
message("\n生成统计摘要...")

summary_stats <- plot_data_all %>%
  group_by(feature, Cohort, Group_label) %>%
  summarise(
    n = n(),
    mean = mean(value, na.rm = TRUE),
    median = median(value, na.rm = TRUE),
    sd = sd(value, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = Group_label,
    values_from = c(n, mean, median, sd),
    names_sep = "_"
  ) %>%
  mutate(
    direction = ifelse(mean_Group.1 > mean_Group.0, "G1 > G0", "G0 > G1")
  )

# 检查方向一致性
direction_check <- summary_stats %>%
  select(feature, Cohort, direction) %>%
  pivot_wider(names_from = Cohort, values_from = direction) %>%
  mutate(
    consistent = Discovery == Validation,
    warning = ifelse(!consistent, "⚠️ REVERSED", "")
  )

summary_stats <- summary_stats %>%
  left_join(direction_check %>% select(feature, consistent, warning), by = "feature")

write.csv(summary_stats, file.path(OUTPUT_DIR, "feature_summary_statistics.csv"), row.names = FALSE)
write.csv(direction_check, file.path(OUTPUT_DIR, "direction_consistency_check.csv"), row.names = FALSE)

message("✓ 统计摘要已保存")

# 打印方向一致性检查结果
message("\n======================================================================")
message("方向一致性检查:")
message("======================================================================")
print(direction_check %>% select(feature, Discovery, Validation, consistent, warning))

message("\n======================================================================")
message("完成!")
message(sprintf("输出目录: %s", OUTPUT_DIR))
message("======================================================================")

