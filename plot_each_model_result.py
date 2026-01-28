#!/usr/bin/env python3
"""
为 multi_seed_summary.csv 中的每个模型配置绘制结果图
风格参考 svm_linear_simple copy.py

作者：Wendy
日期：2026-01-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, 
    classification_report, recall_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except:
    HAS_XGBOOST = False

# ==================== 配置 ====================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# 三个配置的数据和结果文件
CONFIGS = [
    {
        'name': 'no_sweeteners',
        'discovery_file': 'data/Discovery_HK_no_sweeteners.csv',
        'validation_file': 'data/LR_AOCC_MiRES_combined_Validation_cohort_individual_input_table.csv',
        'config_file': 'results_no_sweeteners/v4_multi_model_best_configs.csv',
        'output_dir': 'model_results_no_sweeteners'
    },
    {
        'name': 'sac_suc_asp',
        'discovery_file': 'data/Discovery_HK_sac_suc_asp.csv',
        'validation_file': 'data/Validation_AUS_KM_sac_suc_asp.csv',
        'config_file': 'results_sac_suc_asp/v4_multi_model_best_configs.csv',
        'output_dir': 'model_results_sac_suc_asp'
    },
    {
        'name': 'sac_suc',
        'discovery_file': 'data/Discovery_HK_sac_suc.csv',
        'validation_file': 'data/Validation_AUS_KM_sac_suc.csv',
        'config_file': 'results_sac_suc/v4_multi_model_best_configs.csv',
        'output_dir': 'model_results_sac_suc'
    }
]


def get_model(model_name, seed=RANDOM_SEED, scale_weight=1.0):
    """根据名称返回模型实例"""
    if model_name == 'Logistic (L2)':
        return LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=seed)
    elif model_name == 'Logistic (L1)':
        return LogisticRegression(C=0.5, penalty='l1', solver='saga', 
                                 class_weight='balanced', max_iter=1000, random_state=seed)
    elif model_name == 'SVM Linear':
        return SVC(kernel='linear', C=0.5, class_weight='balanced', probability=True, random_state=seed)
    elif model_name == 'SVM RBF':
        return SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=seed)
    elif model_name == 'Random Forest':
        return RandomForestClassifier(n_estimators=100, max_depth=3, 
                                     class_weight='balanced', random_state=seed, n_jobs=1)
    elif model_name == 'Gradient Boosting':
        return GradientBoostingClassifier(n_estimators=50, max_depth=2, 
                                         learning_rate=0.1, random_state=seed)
    elif model_name == 'XGBoost' and HAS_XGBOOST:
        return XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1,
                            scale_pos_weight=scale_weight, reg_alpha=0.5,
                            random_state=seed, use_label_encoder=False, 
                            eval_metric='logloss', n_jobs=1)
    else:
        return SVC(kernel='linear', C=0.5, class_weight='balanced', probability=True, random_state=seed)


def plot_model_result(model_name, features, X_train, y_train, X_val, y_val, 
                      train_auc_mean, cv_auc_mean, val_auc_mean,
                      train_auc_ci, cv_auc_ci, val_auc_ci, output_dir):
    """
    为单个模型绘制结果图
    """
    # 准备数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 获取模型
    scale_weight = sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1.0
    model = get_model(model_name, RANDOM_SEED, scale_weight)
    
    # 训练
    model.fit(X_train_scaled, y_train)
    
    # 预测
    if hasattr(model, 'predict_proba'):
        y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
        y_prob_val = model.predict_proba(X_val_scaled)[:, 1]
    else:
        y_prob_train = model.decision_function(X_train_scaled)
        y_prob_val = model.decision_function(X_val_scaled)
    
    # 计算 AUC
    train_auc = roc_auc_score(y_train, y_prob_train)
    val_auc = roc_auc_score(y_val, y_prob_val)
    
    # ROC 曲线数据
    fpr, tpr, thresholds = roc_curve(y_val, y_prob_val)
    
    # 最优阈值 (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    # 使用最优阈值预测
    y_pred_val = (y_prob_val >= best_threshold).astype(int)
    
    # 计算指标
    sensitivity = recall_score(y_val, y_pred_val, pos_label=1)
    specificity = recall_score(y_val, y_pred_val, pos_label=0)
    cm = confusion_matrix(y_val, y_pred_val)
    
    # 计算样本统计
    train_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    val_counts = dict(zip(*np.unique(y_val, return_counts=True)))
    train_total = len(y_train)
    val_total = len(y_val)
    train_class0 = train_counts.get(0, 0)
    train_class1 = train_counts.get(1, 0)
    val_class0 = val_counts.get(0, 0)
    val_class1 = val_counts.get(1, 0)
    
    # ==================== 绘图 ====================
    # 根据特征数量调整图表高度
    n_features = len(features)
    extra_height = max(0, (n_features // 6) * 0.3)  # 每6个特征增加0.3英寸高度
    fig, axes = plt.subplots(1, 3, figsize=(15, 5 + extra_height))
    
    # 模型名称简化（用于文件名）
    model_name_safe = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    
    # 1. ROC Curve
    ax1 = axes[0]
    ax1.plot(fpr, tpr, color='#8B5CF6', linewidth=2, label=f'Val AUC = {val_auc:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax1.scatter([fpr[best_idx]], [tpr[best_idx]], color='red', s=100, zorder=5, 
                label=f'Optimal (SEN={sensitivity:.2f}, SPE={specificity:.2f})')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    title1 = f'ROC Curve - {model_name}\n'
    title1 += f'Train: n={train_total} (Class 0: {train_class0}, Class 1: {train_class1}) \n '
    title1 += f'Val: n={val_total} (Class 0: {val_class0}, Class 1: {val_class1})'
    ax1.set_title(title1, fontweight='bold', fontsize=11)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 在 ROC 图下方添加特征列表
    # 格式化特征文本，每行显示约 5-6 个特征
    n_per_line = 6
    features_lines = []
    for i in range(0, len(features), n_per_line):
        features_lines.append(", ".join(features[i:i+n_per_line]))
    
    features_text = f'Selected markers ({len(features)}):\n' + '\n'.join(features_lines)
    
    # 计算需要的垂直空间
    n_lines = len(features_lines) + 1
    y_offset = -0.12 - (n_lines * 0.03)  # 根据行数调整位置
    
    ax1.text(0.5, y_offset, features_text, transform=ax1.transAxes, 
             fontsize=8, ha='center', va='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 2. Confusion Matrix (Validation Set)
    ax2 = axes[1]
    im = ax2.imshow(cm, cmap='Purples')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Pred 0', 'Pred 1'], fontsize=11)
    ax2.set_yticklabels(['True 0', 'True 1'], fontsize=11)
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, cm[i, j], ha='center', va='center', fontsize=16, fontweight='bold',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    title2 = f'Confusion Matrix (Validation Set) - {model_name}\n'
    # title2 += f'Validation: n={val_total} (Class 0: {val_class0}, Class 1: {val_class1})'
    ax2.set_title(title2, fontweight='bold', fontsize=11)
    plt.colorbar(im, ax=ax2)
    
    # 3. Performance Metrics (只显示 AUC，不显示 SEN/SPE)
    ax3 = axes[2]
    metrics = ['Train AUC', 'CV AUC', 'Val AUC']
    # 使用实际计算的值，CV AUC 从配置文件读取（因为需要交叉验证，这里不重新计算）
    values = [train_auc, cv_auc_mean, val_auc]
    cis = [0, cv_auc_ci, 0]  # 只有 CV AUC 有 CI，其他使用实际计算值
    colors = ['#F18F01', '#2E86AB', '#A23B72']
    
    x_pos = np.arange(len(metrics))
    bars = ax3.bar(x_pos, values, yerr=cis, capsize=4, color=colors, 
                   edgecolor='black', linewidth=1, alpha=0.8)
    
    # 数值标注
    for i, (bar, val, ci) in enumerate(zip(bars, values, cis)):
        label = f'{val:.3f}'
        if ci > 0:
            label += f'\n±{ci:.3f}'
        ax3.text(bar.get_x() + bar.get_width()/2, val + ci + 0.03, label, 
                 ha='center', fontsize=10, fontweight='bold', color=colors[i])
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metrics, fontsize=11)
    ax3.set_ylim([0, 1.15])
    ax3.set_ylabel('AUC', fontsize=12)
    ax3.set_title(f'Performance Metrics - {model_name}', fontweight='bold', fontsize=13)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # 为底部特征文本留出空间
    
    # 保存
    output_file = os.path.join(output_dir, f'{model_name_safe}_results.png')
    plt.savefig(output_file, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close()
    
    print(f"  ✓ 已保存: {output_file}")
    
    return {
        'model': model_name,
        'n_features': len(features),
        'train_auc': train_auc,
        'val_auc': val_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'threshold': best_threshold
    }


def process_config(config):
    """
    处理单个配置
    """
    print("\n" + "="*70)
    print(f"   配置: {config['name']}")
    print("="*70)
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 检查配置文件是否存在
    if not os.path.exists(config['config_file']):
        print(f"  ⚠ 跳过: 配置文件不存在 - {config['config_file']}")
        return None
    
    # 读取配置
    config_df = pd.read_csv(config['config_file'])
    print(f"\n加载配置: {config['config_file']}")
    print(f"模型数量: {len(config_df)}")
    
    # 读取数据
    print(f"\n加载数据:")
    print(f"  Discovery: {config['discovery_file']}")
    if not os.path.exists(config['discovery_file']):
        print(f"  ⚠ 跳过: Discovery 文件不存在 - {config['discovery_file']}")
        return None
    
    disc_df = pd.read_csv(config['discovery_file'], na_values=['.', '', ' '])
    disc_df['Group'] = pd.to_numeric(disc_df['Group'], errors='coerce')
    disc_df = disc_df.dropna(subset=['Group'])
    disc_df['Group'] = disc_df['Group'].astype(int)
    
    print(f"  Validation: {config['validation_file']}")
    if not os.path.exists(config['validation_file']):
        print(f"  ⚠ 跳过: Validation 文件不存在 - {config['validation_file']}")
        return None
    
    val_df = pd.read_csv(config['validation_file'], na_values=['.', '', ' '])
    val_df['Group'] = pd.to_numeric(val_df['Group'], errors='coerce')
    val_df = val_df.dropna(subset=['Group'])
    val_df['Group'] = val_df['Group'].astype(int)
    
    y_train = disc_df['Group']
    y_val = val_df['Group']
    
    # 排除非特征列
    exclude_cols = ['Group', 'sampleID', 'cdai']
    X_disc_all = disc_df.drop([c for c in exclude_cols if c in disc_df.columns], axis=1)
    X_val_all = val_df.drop([c for c in exclude_cols if c in val_df.columns], axis=1)
    
    # 转换为数值
    X_disc_all = X_disc_all.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_val_all = X_val_all.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Log1p 变换
    X_disc_all = np.log1p(X_disc_all)
    X_val_all = np.log1p(X_val_all)
    
    print(f"\n  Discovery: {len(X_disc_all)} samples, Group分布: {dict(y_train.value_counts())}")
    print(f"  Validation: {len(X_val_all)} samples, Group分布: {dict(y_val.value_counts())}")
    
    # 为每个模型绘图
    print(f"\n{'='*70}")
    print("开始为每个模型绘制结果图...")
    print(f"{'='*70}")
    
    all_results = []
    
    for idx, row in config_df.iterrows():
        model_name = row['model']
        features = [f.strip() for f in row['features'].split(',')]
        
        # 从配置中获取值（注意：v4_multi_model_best_configs.csv 没有 CI，只有 mean）
        # 尝试读取，如果列名不同则使用备用名称
        try:
            train_auc_mean = row['train_auc'] if 'train_auc' in row else row.get('Train_AUC_mean', 0)
            cv_auc_mean = row['cv_auc'] if 'cv_auc' in row else row.get('CV_AUC_mean', 0)
            val_auc_mean = row['val_auc'] if 'val_auc' in row else row.get('Val_AUC_mean', 0)
        except:
            train_auc_mean = 0
            cv_auc_mean = 0
            val_auc_mean = 0
        
        # 如果没有 CI 列，设为 0
        train_auc_ci = row.get('Train_AUC_CI95', 0) if 'Train_AUC_CI95' in row else 0
        cv_auc_ci = row.get('CV_AUC_CI95', 0) if 'CV_AUC_CI95' in row else 0
        val_auc_ci = row.get('Val_AUC_CI95', 0) if 'Val_AUC_CI95' in row else 0
        
        print(f"\n[{idx+1}/{len(config_df)}] {model_name} (n={len(features)})")
        print(f"  特征: {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")
        
        # 检查特征
        valid_features = [f for f in features if f in X_disc_all.columns and f in X_val_all.columns]
        if len(valid_features) == 0:
            print(f"  ⚠ 跳过: 无有效特征")
            continue
        
        if len(valid_features) < len(features):
            print(f"  ⚠ 警告: 使用 {len(valid_features)}/{len(features)} 个特征")
        
        # 提取特征
        X_train = X_disc_all[valid_features]
        X_val = X_val_all[valid_features]
        
        # 绘图
        result = plot_model_result(
            model_name, valid_features, 
            X_train, y_train, X_val, y_val,
            train_auc_mean, cv_auc_mean, val_auc_mean,
            train_auc_ci, cv_auc_ci, val_auc_ci,
            config['output_dir']
        )
        all_results.append(result)
    
    # 保存汇总
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_path = os.path.join(config['output_dir'], 'all_models_summary.csv')
        results_df.to_csv(output_path, index=False)
        
        print(f"\n{'='*70}")
        print(f"完成！所有结果已保存到 {config['output_dir']}/ 目录")
        print(f"{'='*70}")
        
        # 打印汇总
        print(f"\n{'Model':<20} {'Train AUC':<12} {'Val AUC':<12} {'SEN':<10} {'SPE':<10}")
        print("-"*65)
        for r in all_results:
            print(f"{r['model']:<20} {r['train_auc']:.4f}       {r['val_auc']:.4f}       "
                  f"{r['sensitivity']:.4f}     {r['specificity']:.4f}")
    
    return all_results


def main():
    print("="*70)
    print("   为每个配置的模型绘制结果图")
    print("="*70)
    print(f"\n共 {len(CONFIGS)} 个配置需要处理")
    
    all_config_results = {}
    
    for config in CONFIGS:
        results = process_config(config)
        if results:
            all_config_results[config['name']] = results
    
    print("\n" + "="*70)
    print("   所有配置处理完成！")
    print("="*70)
    print(f"\n成功处理的配置: {list(all_config_results.keys())}")


if __name__ == '__main__':
    main()

