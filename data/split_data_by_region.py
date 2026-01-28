#!/usr/bin/env python3
"""
将 Discovery 数据按区域分割：
- HK 样本 → Discovery set
- AUS + KM 样本 → Validation set
"""

import pandas as pd
import re

# 读取原始数据
input_file = "Discovery_P80_sweetners_CD_individual_activity_sac_suc.csv"
print(f"读取数据: {input_file}")

df = pd.read_csv(input_file, na_values=['.', '', ' '])

# 从 sampleID 提取区域信息
df['Region'] = df['sampleID'].str.replace(r'\d+', '', regex=True)

# 清洗 Group 列
df['Group'] = pd.to_numeric(df['Group'], errors='coerce')
df = df.dropna(subset=['Group'])
df['Group'] = df['Group'].astype(int)

# 统计各区域样本数
print("\n区域统计:")
region_counts = df.groupby(['Region', 'Group']).size().unstack(fill_value=0)
print(region_counts)
print(f"\n总样本数: {len(df)}")

# 分割数据
hk_df = df[df['Region'] == 'HK'].copy()
aus_km_df = df[df['Region'].isin(['AUS', 'KM'])].copy()

# 删除临时添加的 Region 列
hk_df = hk_df.drop('Region', axis=1)
aus_km_df = aus_km_df.drop('Region', axis=1)

# 保存文件
discovery_file = "Discovery_HK_sac_suc.csv"
validation_file = "Validation_AUS_KM_sac_suc.csv"

hk_df.to_csv(discovery_file, index=False)
aus_km_df.to_csv(validation_file, index=False)

