# 诊断模型Pipeline脚本运行逻辑

本文档详细说明三个核心脚本的运行逻辑和工作流程。

---

## 📋 脚本概览

### 1. `diagnostic_model_pipeline_v4.py` - 完整Pipeline
**功能**: 完整的诊断模型构建流程
**输入**: Discovery数据集 + Validation数据集
**输出**: 特征排名、模型对比、最佳模型配置

### 2. `seed_experiment.py` - Seed实验
**功能**: 测试不同随机种子，找到最佳配置
**输入**: Discovery数据集 + Validation数据集
**输出**: 每个seed的结果 + 最佳seed推荐

### 3. `svm_linear_simple.py` - 简单训练验证
**功能**: 使用最佳seed和特征进行最终模型训练
**输入**: 预选特征 + 最佳seed
**输出**: 最终模型性能指标 + 可视化结果

---

## 🔄 完整工作流程

```
┌─────────────────────────────────────────────────┐
│  Step 1: seed_experiment.py                     │
│  ─────────────────────────────────────────────  │
│  • 测试 seeds 1-200                             │
│  • 每个seed独立运行完整pipeline                   │
│  • 记录验证集AUC                                 │
│  • 输出: 最佳seed + 对应特征                      │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Step 2: diagnostic_model_pipeline_v4.py        │
│  ─────────────────────────────────────────────  │
│  • 数据变换对比 (Raw/Log1p/Sqrt/Yeo-Johnson/QN) │
│  • 特征选择 (MW/Effect/L1/RF/MI综合排名)         │
│  • 多模型训练 (LR/SVM/RF/GB/XGB)                 │
│  • 特征数量递增实验                               │
│  • 输出: v4_selected_features.csv                │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Step 3: svm_linear_simple.py                   │
│  ─────────────────────────────────────────────  │
│  • 使用最佳seed和特征                             │
│  • 训练SVM Linear模型                            │
│  • 计算SEN/SPE/阈值                              │
│  • 生成最终可视化                                 │
└─────────────────────────────────────────────────┘
```

---

## 📝 详细脚本说明

### 1️⃣ diagnostic_model_pipeline_v4.py

#### **核心类**: `DiagnosticModelPipelineV4`

#### **主要步骤**:

**Step 1: 数据加载与预处理**
- 读取Discovery和Validation CSV文件
- 清洗Group列（转换为数值，过滤异常值）
- 提取共有特征
- 排除非特征列（sampleID等）

**Step 2: 数据变换对比** (`compare_transformations()`)
- 测试5种变换方法:
  - Raw（原始）
  - Log1p（对数+1）
  - Sqrt（平方根）
  - Yeo-Johnson（幂变换）
  - Quantile-Normal（分位数正态化）
- 使用5折交叉验证评估每种变换
- 自动选择CV AUC最高的变换方法

**Step 3: 应用最佳变换** (`apply_best_transformation()`)
- 对Discovery和Validation数据应用选定的变换
- 确保变换参数只在Discovery上fit，Validation上transform

**Step 4: 特征选择** (`feature_selection()`)
- **5种方法综合评分**:
  1. Mann-Whitney U检验 (权重×2)
  2. Effect Size (权重×2)
  3. L1正则化 (权重×1.5)
  4. Random Forest重要性 (权重×1)
  5. Mutual Information (权重×1)
- 计算综合排名 (`Avg_rank`)
- 选择Top N特征（默认12个）
- 保存: `v4_feature_ranking.csv`, `v4_selected_features.csv`

**Step 5: 模型训练** (`train_models()`)
- 训练7种模型:
  - Logistic Regression (L2)
  - Logistic Regression (L1)
  - SVM Linear
  - SVM RBF
  - Random Forest
  - Gradient Boosting
  - XGBoost (如果可用)
- 5折交叉验证评估
- 在验证集上评估性能
- 自动选择验证集AUC最高的模型

**Step 6: 特征数量递增实验** (`feature_incremental_experiment()`)
- 测试Top 3到Top N特征的不同组合
- 评估不同特征数量对性能的影响
- 保存: `v4_feature_incremental_results.csv`

**Step 7: 多模型对比实验** (`run_multiple_models_incremental()`)
- 对每个模型测试不同特征数量
- 找到每个模型的最佳特征配置
- 保存: `v4_multi_model_best_configs.csv`

**Step 8: 结果可视化** (`plot_results()`)
- 特征排名图
- ROC曲线对比
- 混淆矩阵
- 模型性能对比
- 保存: `v4_02_final_results.png`

#### **关键输出文件**:
- `v4_feature_ranking.csv` - 所有特征的综合排名
- `v4_selected_features.csv` - 选定的Top特征列表
- `v4_model_comparison.csv` - 模型性能对比
- `v4_feature_incremental_results.csv` - 特征数量实验
- `v4_multi_model_best_configs.csv` - 各模型最佳配置
- `v4_01_transformation_comparison.png` - 变换方法对比图
- `v4_02_final_results.png` - 最终结果可视化
- `v4_04_multi_model_comparison.png` - 多模型对比图

---

### 2️⃣ seed_experiment.py

#### **核心功能**: 系统性测试不同随机种子

#### **主要步骤**:

**Step 1: 初始化**
- 设置随机种子范围（默认1-200）
- 创建时间戳命名的输出目录

**Step 2: 对每个Seed运行Pipeline** (`run_pipeline_with_seed()`)
- **数据加载**:
  - 读取Discovery和Validation数据
  - 清洗Group列
  - 提取共有特征

- **数据变换**:
  - 应用Log1p变换（经验上最佳）

- **特征选择**（与pipeline相同）:
  - Mann-Whitney U
  - Effect Size
  - L1正则化
  - Random Forest
  - Mutual Information
  - 综合排名，选择Top 12特征

- **模型训练**:
  - 训练7种模型
  - 5折交叉验证
  - 验证集评估
  - 记录最佳模型和AUC

- **保存结果**:
  - `seed_XXX/model_results.csv`
  - `seed_XXX/feature_ranking.csv`
  - `seed_XXX/selected_features.csv`

**Step 3: 汇总分析**
- 收集所有seed的结果
- 按验证集AUC排序
- 生成汇总表: `seed_experiment_summary.csv`
- 可视化:
  - AUC分布直方图
  - AUC随seed变化曲线
  - 最佳模型频次
  - Top 10 seeds柱状图

**Step 4: 输出最佳配置**
- 识别验证集AUC最高的seed
- 输出该seed的:
  - 最佳模型名称
  - 验证集AUC
  - CV AUC
  - Sensitivity/Specificity
  - 选定的特征列表

#### **关键输出文件**:
- `seed_experiment_YYYYMMDD_HHMMSS/` - 主目录
  - `seed_experiment_summary.csv` - 所有seed汇总
  - `seed_experiment_summary.png` - 可视化汇总
  - `seed_001` 到 `seed_200` - 每个seed的详细结果

#### **使用场景**:
- 当需要找到最稳定的模型配置时
- 当发现模型性能对随机种子敏感时
- 当需要确保结果可复现时

---

### 3️⃣ svm_linear_simple.py

#### **核心功能**: 使用最佳配置进行最终模型训练

#### **主要步骤**:

**Step 1: 配置加载**
- 设置最佳随机种子（从seed_experiment结果获得）
- 加载预选特征列表（从pipeline或seed_experiment获得）

**Step 2: 数据加载**
- 读取Discovery和Validation数据
- 提取选定特征
- 应用Log1p变换
- StandardScaler标准化（仅在Discovery上fit）

**Step 3: 模型训练**
- 使用SVM Linear模型:
  - `kernel='linear'`
  - `C=0.5`
  - `class_weight='balanced'`
  - `probability=True`
- 在Discovery集上训练

**Step 4: Discovery集评估**
- 计算训练集AUC
- 生成分类报告

**Step 5: Validation集评估**
- 预测验证集概率
- 计算验证集AUC
- **阈值优化** (Youden's J统计量):
  - 找到最大化 `Sensitivity + Specificity - 1` 的阈值
- 使用最优阈值计算:
  - Sensitivity (召回率，正类)
  - Specificity (召回率，负类)
- 生成混淆矩阵

**Step 6: 结果可视化**
- **3个子图**:
  1. ROC曲线（标注最优阈值点）
  2. 混淆矩阵热图
  3. 性能指标柱状图（AUC/SEN/SPE）
- 保存: `svm_linear_results.png`

**Step 7: 结果保存**
- 保存CSV: `svm_linear_results.csv`
  - Train_AUC
  - Val_AUC
  - Sensitivity
  - Specificity
  - Threshold

#### **关键输出文件**:
- `svm_linear_results.png` - 可视化结果
- `svm_linear_results.csv` - 性能指标

#### **使用场景**:
- 最终模型验证和报告
- 生成发表用的图表
- 临床应用的模型部署准备

---

## ⚠️ 重要注意事项

### 数据泄露防护

✅ **正确做法**:
- StandardScaler只在Discovery上`fit_transform()`，在Validation上`transform()`
- 数据变换参数只在Discovery上学习
- 特征选择只在Discovery上进行
- 模型只在Discovery上训练

❌ **错误做法**:
- 在包含Validation的数据上fit scaler
- 使用Validation数据选择特征
- 在Validation上训练模型

### 特征一致性检查

在使用新数据集时，需要检查:
1. **方向一致性**: 特征在Discovery和Validation上的组间差异方向是否一致
2. **分布相似性**: 两个cohort的特征分布是否相似
3. **缺失值处理**: 确保缺失值处理方式一致

### 推荐运行顺序

```bash
# 1. 先运行seed实验找到最佳配置
python seed_experiment.py

# 2. 运行完整pipeline（可选，用于深入分析）
python diagnostic_model_pipeline_v4.py

# 3. 使用最佳配置进行最终训练
python svm_linear_simple.py
```

---

## 📊 性能指标说明

- **AUC (Area Under ROC Curve)**: 模型整体区分能力，0.5=随机，1.0完美
- **Sensitivity (敏感性)**: 真阳性率，正确识别病人的比例
- **Specificity (特异性)**: 真阴性率，正确识别健康人的比例
- **Threshold (阈值)**: 用于二分类的概率阈值，通过Youden's J优化

---

*最后更新: 2026-01-26*

