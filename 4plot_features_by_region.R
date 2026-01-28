#!/usr/bin/env Rscript
# 按区域绘制 Discovery cohort 中 Sac_S, Suc_S 和 cdai 特征的箱线图
# 基于 plot_top_features_boxplot.R 的绘图风格

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
OUTPUT_DIR <- "feature_boxplot_results/features_by_region"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# 目标特征
TARGET_FEATURES <- c("Sac_S", "Suc_S", "cdai")

# 颜色定义
GROUP0_COLOR <- "#2E86AB"  # 蓝色 - Group 0 (健康/对照)
GROUP1_COLOR <- "#A23B72"  # 紫色 - Group 1 (病人)
REGION_COLORS <- c("AUS" = "#8B5CF6", "HK" = "#F18F01", "KM" = "#C73E1D")

# ============================================================================
# 读取数据
# ============================================================================
message("读取数据...")

# 读取Discovery数据
disc_df <- read.csv("data/Discovery_P80_sweetners_CD_individual_activity(1).csv", 
                    na.strings = c('.', '', ' '), stringsAsFactors = FALSE)

# 清洗 Group 列
disc_df$Group <- as.numeric(disc_df$Group)
disc_df <- disc_df %>% filter(!is.na(Group))
disc_df$Group <- as.integer(disc_df$Group)

# 从 sampleID 提取区域信息
disc_df$Region <- gsub("[0-9]", "", disc_df$sampleID)
disc_df$Region <- factor(disc_df$Region, levels = c("AUS", "HK", "KM"))

message(sprintf("\n数据加载完成:"))
message(sprintf("  总样本数: %d", nrow(disc_df)))
message(sprintf("  Group 0: %d, Group 1: %d", 
                sum(disc_df$Group == 0), 
                sum(disc_df$Group == 1)))

# 按区域统计
region_stats <- disc_df %>%
  group_by(Region, Group) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(names_from = Group, values_from = n, names_prefix = "Group_")

message("\n按区域统计:")
print(region_stats)

# ============================================================================
# 准备绘图数据
# ============================================================================
message("\n准备绘图数据...")

plot_data_list <- list()

for (feat in TARGET_FEATURES) {
  # 检查特征是否存在
  if (!(feat %in% colnames(disc_df))) {
    message(sprintf("  跳过 %s (不存在于数据中)", feat))
    next
  }
  
  # 提取特征数据
  feat_data <- disc_df %>%
    select(sampleID, Region, Group, all_of(feat)) %>%
    rename(value = all_of(feat)) %>%
    mutate(value = as.numeric(value)) %>%
    filter(!is.na(value)) %>%
    mutate(
      feature = feat,
      Group_label = ifelse(Group == 0, "Group 0", "Group 1"),
      Group_label = factor(Group_label, levels = c("Group 0", "Group 1"))
    )
  
  # 对数转换（处理0值和负值）
  if (feat == "cdai") {
    # cdai 使用原始值（不转换）
    feat_data$log_value <- feat_data$value
    feat_data$display_value <- feat_data$value  # 原始值用于显示
  } else {
    # Sac_S 和 Suc_S 使用对数转换
    feat_data$log_value <- log10(ifelse(feat_data$value == 0, 1e-6, feat_data$value))
    feat_data$display_value <- feat_data$value
  }
  
  # 创建统一的 y 值列（用于组合图）
  feat_data$plot_value <- if (feat == "cdai") feat_data$value else feat_data$log_value
  
  plot_data_list[[feat]] <- feat_data
  
  message(sprintf("  ✓ %s: %d 个样本", feat, nrow(feat_data)))
}

plot_data_all <- bind_rows(plot_data_list)

# ============================================================================
# 绘制箱线图 - 按区域分组
# ============================================================================
message("\n绘制箱线图...")

boxplot_dir <- file.path(OUTPUT_DIR, "individual_features")
dir.create(boxplot_dir, showWarnings = FALSE)

for (feat in names(plot_data_list)) {
  feat_data <- plot_data_list[[feat]]
  
  # 计算统计信息（按区域和组）
  stats_by_region <- feat_data %>%
    group_by(Region, Group_label) %>%
    summarise(
      n = n(),
      mean = mean(display_value, na.rm = TRUE),
      median = median(display_value, na.rm = TRUE),
      sd = sd(display_value, na.rm = TRUE),
      .groups = "drop"
    )
  
  # 计算每个区域的方向（G1 vs G0）
  direction_by_region <- stats_by_region %>%
    select(Region, Group_label, mean) %>%
    pivot_wider(names_from = Group_label, values_from = mean) %>%
    mutate(
      direction = ifelse(`Group 1` > `Group 0`, "G1 > G0", "G0 > G1"),
      diff = abs(`Group 1` - `Group 0`)
    )
  
  # 创建统计信息标签
  stats_label <- paste(
    sprintf("Region | Direction | Mean Diff"),
    paste(
      sprintf("%s | %s | %.2f", 
              direction_by_region$Region,
              direction_by_region$direction,
              direction_by_region$diff),
      collapse = "\n"
    ),
    sep = "\n"
  )
  
  # 选择 y 轴变量（cdai 使用原始值，其他使用对数）
  y_var <- if (feat == "cdai") "display_value" else "log_value"
  y_label <- if (feat == "cdai") "Value" else "Value (log10)"
  
  # 绘制箱线图
  p <- ggplot(feat_data, aes(x = Region, y = !!sym(y_var), fill = Group_label)) +
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
      title = sprintf("%s by Region", feat),
      subtitle = "Discovery Cohort",
      x = "Region",
      y = y_label,
      fill = "Group"
    ) +
    # 主题
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(face = "italic", size = 12, hjust = 0.5),
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
    color = "black",
    fontface = "plain"
  )
  
  # 保存图片
  filename <- gsub("[^A-Za-z0-9_]", "_", feat)
  output_file <- file.path(boxplot_dir, paste0("boxplot_", filename, "_by_region.png"))
  
  ggsave(
    output_file,
    p,
    width = 10,
    height = 6,
    dpi = 300,
    bg = "white"
  )
  
  message(sprintf("  ✓ %s 已保存", feat))
}

message(sprintf("\n✓ 箱线图已保存到: %s", boxplot_dir))

# ============================================================================
# 绘制组合图 - 三个特征的对比
# ============================================================================
message("\n绘制组合对比图...")

# 为组合图准备数据，为每个特征创建合适的 y 值
plot_data_combined <- plot_data_all %>%
  mutate(
    plot_y = case_when(
      feature == "cdai" ~ display_value,
      TRUE ~ log_value
    ),
    y_label = case_when(
      feature == "cdai" ~ "Value",
      TRUE ~ "Value (log10)"
    )
  )

p_combined <- ggplot(plot_data_combined, aes(x = Region, y = plot_y, fill = Group_label)) +
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
    size = 2,
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
  facet_wrap(~ feature, scales = "free_y", ncol = 3) +
  labs(
    title = "Sac_S, Suc_S, and cdai by Region",
    subtitle = "Discovery Cohort - Comparison Across Regions",
    x = "Region",
    y = "Value",
    fill = "Group"
  ) +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(face = "italic", size = 12, hjust = 0.5),
    axis.text = element_text(color = "black", size = 11),
    axis.title = element_text(size = 13),
    strip.text = element_text(face = "bold", size = 12),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 1),
    legend.position = "right"
  )

ggsave(
  file.path(OUTPUT_DIR, "boxplot_all_features_by_region.png"),
  p_combined,
  width = 14,
  height = 5,
  dpi = 300,
  bg = "white"
)

message("✓ 组合对比图已保存")

# ============================================================================
# 生成统计摘要表
# ============================================================================
message("\n生成统计摘要...")

summary_stats <- plot_data_all %>%
  group_by(feature, Region, Group_label) %>%
  summarise(
    n = n(),
    mean = mean(display_value, na.rm = TRUE),
    median = median(display_value, na.rm = TRUE),
    sd = sd(display_value, na.rm = TRUE),
    min = min(display_value, na.rm = TRUE),
    max = max(display_value, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(feature, Region, Group_label)

# 计算组间差异
group_diff <- summary_stats %>%
  select(feature, Region, Group_label, mean) %>%
  pivot_wider(names_from = Group_label, values_from = mean) %>%
  mutate(
    mean_diff = `Group 1` - `Group 0`,
    direction = ifelse(mean_diff > 0, "G1 > G0", "G0 > G1"),
    abs_diff = abs(mean_diff)
  ) %>%
  arrange(feature, Region)

write.csv(summary_stats, file.path(OUTPUT_DIR, "feature_summary_by_region.csv"), row.names = FALSE)
write.csv(group_diff, file.path(OUTPUT_DIR, "group_difference_by_region.csv"), row.names = FALSE)

message("✓ 统计摘要已保存")

# 打印组间差异
message("\n======================================================================")
message("组间差异 (Group 1 - Group 0) by Region:")
message("======================================================================")
print(group_diff)

message("\n======================================================================")
message("完成!")
message(sprintf("输出目录: %s", OUTPUT_DIR))
message("======================================================================")

