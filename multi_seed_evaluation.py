#!/usr/bin/env python3
"""
多种子评估脚本
基于 v4_multi_model_best_configs.csv 中的最优配置，
使用多个随机种子测试模型稳定性，绘制带 CI 的 AUC 曲线

作者：Wendy
日期：2026-01-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import random
import os

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except:
    HAS_XGBOOST = False

# 绘图设置
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 颜色方案
COLORS = {
    'train': '#F18F01',    # 橙色 - Train AUC
    'cv': '#2E86AB',       # 蓝色 - CV AUC
    'val': '#A23B72',      # 紫色 - Val AUC
}


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_model(model_name, seed, scale_weight=1.0):
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


def evaluate_with_seed(model_name, features, X_disc, y_disc, X_val, y_val, seed):
    """使用指定种子评估模型"""
    set_seed(seed)
    
    # 准备数据
    X_train = X_disc[features]
    X_test = X_val[features]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 计算 scale_weight
    scale_weight = sum(y_disc == 0) / sum(y_disc == 1) if sum(y_disc == 1) > 0 else 1.0
    
    # 获取模型
    model = get_model(model_name, seed, scale_weight)
    
    # CV 评估
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(model, X_train_scaled, y_disc, cv=cv, scoring='roc_auc', n_jobs=1)
    cv_auc = cv_scores.mean()
    
    # 训练模型
    model.fit(X_train_scaled, y_disc)
    
    # 预测
    if hasattr(model, 'predict_proba'):
        y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
        y_prob_val = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob_train = model.decision_function(X_train_scaled)
        y_prob_val = model.decision_function(X_test_scaled)
    
    train_auc = roc_auc_score(y_disc, y_prob_train)
    val_auc = roc_auc_score(y_val, y_prob_val)
    
    return train_auc, cv_auc, val_auc


def run_multi_seed_evaluation(config_file, discovery_file, validation_file, 
                               seeds=range(1, 51), output_prefix='multi_seed'):
    """
    对配置文件中的每个模型进行多种子评估
    
    Args:
        config_file: v4_multi_model_best_configs.csv 路径
        discovery_file: Discovery 数据路径
        validation_file: Validation 数据路径
        seeds: 要测试的种子范围
        output_prefix: 输出文件前缀
    """
    print("="*70)
    print("   Multi-Seed Model Evaluation")
    print("="*70)
    
    # 读取配置
    config_df = pd.read_csv(config_file)
    print(f"\n加载配置: {len(config_df)} 个模型")
    
    # 读取数据
    print(f"加载 Discovery 数据: {discovery_file}")
    disc_df = pd.read_csv(discovery_file, na_values=['.', '', ' '])
    disc_df['Group'] = pd.to_numeric(disc_df['Group'], errors='coerce')
    disc_df = disc_df.dropna(subset=['Group'])
    disc_df['Group'] = disc_df['Group'].astype(int)
    
    print(f"加载 Validation 数据: {validation_file}")
    val_df = pd.read_csv(validation_file, na_values=['.', '', ' '])
    val_df['Group'] = pd.to_numeric(val_df['Group'], errors='coerce')
    val_df = val_df.dropna(subset=['Group'])
    val_df['Group'] = val_df['Group'].astype(int)
    
    y_disc = disc_df['Group']
    y_val = val_df['Group']
    
    # 排除非特征列
    exclude_cols = ['Group', 'sampleID', 'cdai']
    X_disc = disc_df.drop([c for c in exclude_cols if c in disc_df.columns], axis=1)
    X_val = val_df.drop([c for c in exclude_cols if c in val_df.columns], axis=1)
    
    # 转换为数值
    X_disc = X_disc.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_val = X_val.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Log1p 变换
    X_disc = np.log1p(X_disc)
    X_val = np.log1p(X_val)
    
    print(f"\nDiscovery: {len(X_disc)} samples, Group分布: {dict(y_disc.value_counts())}")
    print(f"Validation: {len(X_val)} samples, Group分布: {dict(y_val.value_counts())}")
    print(f"\n测试种子范围: {min(seeds)} - {max(seeds)} (共 {len(list(seeds))} 个)")
    
    # 存储结果
    all_results = {}
    
    for _, row in config_df.iterrows():
        model_name = row['model']
        features = [f.strip() for f in row['features'].split(',')]
        n_features = row['n_features']
        
        # 检查特征是否存在
        valid_features = [f for f in features if f in X_disc.columns and f in X_val.columns]
        if len(valid_features) < len(features):
            print(f"\n警告: {model_name} 部分特征不存在，使用 {len(valid_features)}/{len(features)} 个特征")
        
        if len(valid_features) == 0:
            print(f"\n跳过 {model_name}: 无有效特征")
            continue
        
        print(f"\n{'='*50}")
        print(f"模型: {model_name} (Top{n_features})")
        print(f"特征: {', '.join(valid_features[:5])}...")
        print(f"{'='*50}")
        
        train_aucs = []
        cv_aucs = []
        val_aucs = []
        
        for seed in seeds:
            try:
                train_auc, cv_auc, val_auc = evaluate_with_seed(
                    model_name, valid_features, X_disc, y_disc, X_val, y_val, seed
                )
                train_aucs.append(train_auc)
                cv_aucs.append(cv_auc)
                val_aucs.append(val_auc)
                
                if seed % 10 == 0:
                    print(f"  Seed {seed:3d}: Train={train_auc:.3f}, CV={cv_auc:.3f}, Val={val_auc:.3f}")
            except Exception as e:
                print(f"  Seed {seed}: 错误 - {e}")
        
        all_results[model_name] = {
            'n_features': n_features,
            'features': valid_features,
            'train_aucs': train_aucs,
            'cv_aucs': cv_aucs,
            'val_aucs': val_aucs
        }
        
        # 打印统计摘要
        print(f"\n  统计摘要:")
        print(f"    Train AUC: {np.mean(train_aucs):.4f} ± {np.std(train_aucs):.4f}")
        print(f"    CV AUC:    {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
        print(f"    Val AUC:   {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
    
    # 绘制结果
    plot_multi_seed_results(all_results, output_prefix)
    
    # 保存详细结果
    save_detailed_results(all_results, output_prefix)
    
    return all_results


def plot_multi_seed_results(all_results, output_prefix):
    """绘制带 CI 的 AUC 曲线"""
    n_models = len(all_results)
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ========== 图1: 各模型 Train/CV/Val AUC 对比（带 CI）==========
    ax1 = axes[0]
    
    models = list(all_results.keys())
    x = np.arange(len(models))
    width = 0.25
    
    train_means = [np.mean(all_results[m]['train_aucs']) for m in models]
    train_cis = [1.96 * np.std(all_results[m]['train_aucs']) / np.sqrt(len(all_results[m]['train_aucs'])) for m in models]
    
    cv_means = [np.mean(all_results[m]['cv_aucs']) for m in models]
    cv_cis = [1.96 * np.std(all_results[m]['cv_aucs']) / np.sqrt(len(all_results[m]['cv_aucs'])) for m in models]
    
    val_means = [np.mean(all_results[m]['val_aucs']) for m in models]
    val_cis = [1.96 * np.std(all_results[m]['val_aucs']) / np.sqrt(len(all_results[m]['val_aucs'])) for m in models]
    
    ax1.bar(x - width, train_means, width, yerr=train_cis, capsize=3,
            label='Train AUC', color=COLORS['train'], alpha=0.8)
    ax1.bar(x, cv_means, width, yerr=cv_cis, capsize=3,
            label='CV AUC', color=COLORS['cv'], alpha=0.8)
    ax1.bar(x + width, val_means, width, yerr=val_cis, capsize=3,
            label='Val AUC', color=COLORS['val'], alpha=0.8)
    
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{m}\n(n={all_results[m]['n_features']})" for m in models], 
                        fontsize=9, rotation=15, ha='right')
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('Train / CV / Validation AUC with 95% CI', fontweight='bold', fontsize=13)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim([0.4, 1.05])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标注
    for i, m in enumerate(models):
        ax1.text(i - width, train_means[i] + train_cis[i] + 0.02, f'{train_means[i]:.2f}', 
                ha='center', fontsize=8, color=COLORS['train'])
        ax1.text(i, cv_means[i] + cv_cis[i] + 0.02, f'{cv_means[i]:.2f}', 
                ha='center', fontsize=8, color=COLORS['cv'])
        ax1.text(i + width, val_means[i] + val_cis[i] + 0.02, f'{val_means[i]:.2f}', 
                ha='center', fontsize=8, color=COLORS['val'])
    
    # ========== 图2: 箱线图展示分布 ==========
    ax2 = axes[1]
    
    # 准备数据
    plot_data = []
    for m in models:
        for auc_type, aucs, color in [
            ('Train', all_results[m]['train_aucs'], COLORS['train']),
            ('CV', all_results[m]['cv_aucs'], COLORS['cv']),
            ('Val', all_results[m]['val_aucs'], COLORS['val'])
        ]:
            for auc in aucs:
                plot_data.append({
                    'Model': m,
                    'Type': auc_type,
                    'AUC': auc
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    # 自定义颜色
    palette = {'Train': COLORS['train'], 'CV': COLORS['cv'], 'Val': COLORS['val']}
    
    sns.boxplot(data=plot_df, x='Model', y='AUC', hue='Type', ax=ax2, palette=palette)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax2.set_xticklabels([f"{m}\n(n={all_results[m]['n_features']})" for m in models], 
                        fontsize=9, rotation=15, ha='right')
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_title('AUC Distribution Across Seeds (Boxplot)', fontweight='bold', fontsize=13)
    ax2.legend(title='AUC Type', loc='upper right', fontsize=10)
    ax2.set_ylim([0.4, 1.05])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_auc_comparison.png', bbox_inches='tight', facecolor='white', dpi=150)
    plt.show()
    
    print(f"\n✓ 图表已保存: {output_prefix}_auc_comparison.png")


def save_detailed_results(all_results, output_prefix):
    """保存详细结果到 CSV"""
    summary_data = []
    
    for model_name, data in all_results.items():
        train_aucs = data['train_aucs']
        cv_aucs = data['cv_aucs']
        val_aucs = data['val_aucs']
        
        summary_data.append({
            'Model': model_name,
            'n_features': data['n_features'],
            'Train_AUC_mean': np.mean(train_aucs),
            'Train_AUC_std': np.std(train_aucs),
            'Train_AUC_CI95': 1.96 * np.std(train_aucs) / np.sqrt(len(train_aucs)),
            'CV_AUC_mean': np.mean(cv_aucs),
            'CV_AUC_std': np.std(cv_aucs),
            'CV_AUC_CI95': 1.96 * np.std(cv_aucs) / np.sqrt(len(cv_aucs)),
            'Val_AUC_mean': np.mean(val_aucs),
            'Val_AUC_std': np.std(val_aucs),
            'Val_AUC_CI95': 1.96 * np.std(val_aucs) / np.sqrt(len(val_aucs)),
            'features': ', '.join(data['features'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Val_AUC_mean', ascending=False)
    summary_df.to_csv(f'{output_prefix}_summary.csv', index=False)
    
    print(f"✓ 结果已保存: {output_prefix}_summary.csv")
    
    # 打印摘要表
    print("\n" + "="*70)
    print("   Multi-Seed Evaluation Summary")
    print("="*70)
    print(f"\n{'Model':<20} {'Train AUC':<18} {'CV AUC':<18} {'Val AUC':<18}")
    print("-"*75)
    for _, row in summary_df.iterrows():
        print(f"{row['Model']:<20} "
              f"{row['Train_AUC_mean']:.3f}±{row['Train_AUC_std']:.3f} "
              f"       {row['CV_AUC_mean']:.3f}±{row['CV_AUC_std']:.3f} "
              f"       {row['Val_AUC_mean']:.3f}±{row['Val_AUC_std']:.3f}")


# ==================== 主程序 ====================
if __name__ == '__main__':
    # 配置参数
    CONFIG_FILE = 'v4_multi_model_best_configs.csv'
    DISCOVERY_FILE = 'data/Discovery_HK_sac_suc_asp.csv'
    VALIDATION_FILE = 'data/Validation_AUS_KM_sac_suc_asp.csv'
    SEEDS = range(1, 101)  # 50 个种子
    OUTPUT_PREFIX = 'multi_seed'
    
    # 运行评估
    results = run_multi_seed_evaluation(
        config_file=CONFIG_FILE,
        discovery_file=DISCOVERY_FILE,
        validation_file=VALIDATION_FILE,
        seeds=SEEDS,
        output_prefix=OUTPUT_PREFIX
    )

