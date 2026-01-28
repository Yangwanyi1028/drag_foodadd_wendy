#!/usr/bin/env python3
"""
从 CD_P80_activity_individual.csv 提取 HK 样本作为 Discovery 集
标注为没有 sac, suc, asp（这些列设为 0）

作者：Wendy
日期：2026-01-28
"""

import pandas as pd
import re

# 读取数据
input_file = "CD_P80_activity_individual.csv"
output_file = "Discovery_HK_no_sweeteners.csv"

print(f"读取数据: {input_file}")
df = pd.read_csv(input_file, na_values=['.', '', ' '])

# 提取 HK 样本
print("\n提取 HK 样本...")
hk_df = df[df['sampleID'].str.startswith('HK', na=False)].copy()
print(f"  HK 样本数: {len(hk_df)}")

# 将 Activity 转换为 Group
print("\n转换 Activity 为 Group...")
# CD (CDAI<150) -> 0, CD (CDAI>150) -> 1
hk_df['Group'] = hk_df['Activity'].apply(
    lambda x: 0 if 'CDAI<150' in str(x) else (1 if 'CDAI>150' in str(x) else None)
)
hk_df = hk_df.dropna(subset=['Group'])
hk_df['Group'] = hk_df['Group'].astype(int)

print(f"  Group 分布:")
print(hk_df['Group'].value_counts().sort_index())

# 添加 sac, suc, asp 列（设为 0，表示没有这些甜味剂）
print("\n添加甜味剂列（设为 0）...")
hk_df['Sac_S'] = 0
hk_df['Suc_S'] = 0
hk_df['Asp_S'] = 0

# 删除 Activity 列（已转换为 Group）
hk_df = hk_df.drop('Activity', axis=1)

# 重新排列列，将 Group 放在 sampleID 之后
cols = ['sampleID', 'Group'] + [c for c in hk_df.columns if c not in ['sampleID', 'Group']]
hk_df = hk_df[cols]

# 保存
hk_df.to_csv(output_file, index=False)
print(f"\n✓ 已保存到: {output_file}")
print(f"  总样本数: {len(hk_df)}")
print(f"  总列数: {len(hk_df.columns)}")
print(f"  包含列: sampleID, Group, Sac_S, Suc_S, Asp_S, ...")

