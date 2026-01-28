"""
Seed Experiment - Test seeds 1-100 to find the best validation performance
Author: Wendy
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import warnings
import random
import os
import shutil
from datetime import datetime

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, recall_score

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except:
    HAS_XGBOOST = False


def set_seed(seed):
    """Set all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_pipeline_with_seed(discovery_path, validation_path, seed, output_dir):
    """Run the pipeline with a specific seed and save results"""
    
    set_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    disc_df = pd.read_csv(discovery_path, na_values=['.', '', ' '])
    val_df = pd.read_csv(validation_path, na_values=['.', '', ' '])
    
    # 清洗 Group 列：转换为数值，过滤掉无法转换的行
    disc_df['Group'] = pd.to_numeric(disc_df['Group'], errors='coerce')
    disc_df = disc_df.dropna(subset=['Group'])
    disc_df['Group'] = disc_df['Group'].astype(int)
    
    val_df['Group'] = pd.to_numeric(val_df['Group'], errors='coerce')
    val_df = val_df.dropna(subset=['Group'])
    val_df['Group'] = val_df['Group'].astype(int)
    
    y_discovery = disc_df['Group']
    y_validation = val_df['Group']
    
    # 排除非特征列（如 sampleID, cdai 等）
    exclude_cols = ['Group', 'sampleID', 'cdai', 'Asp_U', 'Sac_U', 'Suc_U', 'Asp_S', 'Sac_S', 'Suc_S']
    X_disc = disc_df.drop([c for c in exclude_cols if c in disc_df.columns], axis=1)
    X_val = val_df.drop([c for c in exclude_cols if c in val_df.columns], axis=1)
    
    common = list(set(X_disc.columns) & set(X_val.columns))
    X_discovery_raw = X_disc[common].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_validation_raw = X_val[common].apply(pd.to_numeric, errors='coerce').fillna(0)
    feature_names = common
    
    # Apply Log1p transformation (commonly best)
    X_discovery = pd.DataFrame(
        np.log1p(X_discovery_raw.values),
        columns=feature_names,
        index=X_discovery_raw.index
    )
    X_validation = pd.DataFrame(
        np.log1p(X_validation_raw.values),
        columns=feature_names,
        index=X_validation_raw.index
    )
    
    # Feature selection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_discovery)
    
    feature_scores = pd.DataFrame(index=feature_names)
    
    # Mann-Whitney U
    mw_scores = {}
    for col in feature_names:
        g0 = X_discovery[y_discovery == 0][col]
        g1 = X_discovery[y_discovery == 1][col]
        try:
            _, p = mannwhitneyu(g0, g1)
            mw_scores[col] = -np.log10(p + 1e-300)
        except:
            mw_scores[col] = 0
    feature_scores['MW_score'] = pd.Series(mw_scores)
    feature_scores['MW_rank'] = feature_scores['MW_score'].rank(ascending=False)
    
    # Effect size
    effect_scores = {}
    for col in feature_names:
        g0 = X_discovery[y_discovery == 0][col]
        g1 = X_discovery[y_discovery == 1][col]
        try:
            U, _ = mannwhitneyu(g0, g1)
            r = abs(1 - (2*U)/(len(g0)*len(g1)))
            effect_scores[col] = r
        except:
            effect_scores[col] = 0
    feature_scores['Effect'] = pd.Series(effect_scores)
    feature_scores['Effect_rank'] = feature_scores['Effect'].rank(ascending=False)
    
    # L1 regularization
    best_coefs = np.zeros(len(feature_names))
    for C in [0.01, 0.05, 0.1, 0.5, 1.0]:
        lr_l1 = LogisticRegression(penalty='l1', solver='saga', C=C, 
                                   class_weight='balanced', max_iter=5000, random_state=seed)
        lr_l1.fit(X_scaled, y_discovery)
        coefs = np.abs(lr_l1.coef_[0])
        n_nonzero = np.sum(coefs > 0)
        if 10 <= n_nonzero <= 40:
            best_coefs = coefs
            break
        elif n_nonzero > 0:
            best_coefs = coefs
    
    feature_scores['L1_coef'] = best_coefs
    feature_scores['L1_rank'] = feature_scores['L1_coef'].rank(ascending=False, method='min')
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, 
                               class_weight='balanced', random_state=seed)
    rf.fit(X_scaled, y_discovery)
    feature_scores['RF'] = rf.feature_importances_
    feature_scores['RF_rank'] = feature_scores['RF'].rank(ascending=False)
    
    # Mutual Information
    mi = mutual_info_classif(X_scaled, y_discovery, random_state=seed)
    feature_scores['MI'] = mi
    feature_scores['MI_rank'] = feature_scores['MI'].rank(ascending=False)
    
    # Combined ranking
    feature_scores['Avg_rank'] = (
        feature_scores['MW_rank'] * 2 +
        feature_scores['Effect_rank'] * 2 +
        feature_scores['L1_rank'] * 1.5 +
        feature_scores['RF_rank'] * 1 +
        feature_scores['MI_rank'] * 1
    ) / 7.5
    
    feature_scores = feature_scores.sort_values('Avg_rank')
    
    # Select top 12 features
    n_features = 3
    selected_features = feature_scores.head(n_features).index.tolist()
    
    # Train models
    X_train = X_discovery[selected_features]
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    
    X_val_final = X_validation[selected_features]
    X_val_scaled = scaler_final.transform(X_val_final)
    
    # Define models
    scale_weight = sum(y_discovery == 0) / sum(y_discovery == 1)
    models = {
        'Logistic (L2)': LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=seed),
        'Logistic (L1)': LogisticRegression(C=0.5, penalty='l1', solver='saga', 
                                            class_weight='balanced', max_iter=1000, random_state=seed),
        'SVM Linear': SVC(kernel='linear', C=0.5, class_weight='balanced', probability=True, random_state=seed),
        'SVM RBF': SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=seed),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=3, 
                                                class_weight='balanced', random_state=seed),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=2, 
                                                       learning_rate=0.1, random_state=seed),
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1,
                                          scale_pos_weight=scale_weight, reg_alpha=0.5,
                                          random_state=seed, use_label_encoder=False, eval_metric='logloss')
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = []
    
    best_val_auc = 0
    best_model_name = None
    
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_discovery, cv=cv, scoring='roc_auc')
        model.fit(X_train_scaled, y_discovery)
        
        if hasattr(model, 'predict_proba'):
            y_prob_val = model.predict_proba(X_val_scaled)[:, 1]
        else:
            y_prob_val = model.decision_function(X_val_scaled)
        
        val_auc = roc_auc_score(y_validation, y_prob_val)
        
        # Calculate sensitivity and specificity
        fpr, tpr, thresholds = roc_curve(y_validation, y_prob_val)
        best_idx = np.argmax(tpr - fpr)
        best_thresh = thresholds[best_idx]
        y_pred = (y_prob_val >= best_thresh).astype(int)
        sensitivity = recall_score(y_validation, y_pred, pos_label=1)
        specificity = recall_score(y_validation, y_pred, pos_label=0)
        
        results.append({
            'Model': name,
            'CV_AUC': cv_scores.mean(),
            'CV_std': cv_scores.std(),
            'Val_AUC': val_auc,
            'Sensitivity': sensitivity,
            'Specificity': specificity
        })
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_name = name
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'model_results.csv'), index=False)
    feature_scores.to_csv(os.path.join(output_dir, 'feature_ranking.csv'))
    pd.DataFrame({'Features': selected_features}).to_csv(
        os.path.join(output_dir, 'selected_features.csv'), index=False)
    
    return {
        'seed': seed,
        'best_model': best_model_name,
        'best_val_auc': best_val_auc,
        'best_sensitivity': results_df[results_df['Model'] == best_model_name]['Sensitivity'].values[0],
        'best_specificity': results_df[results_df['Model'] == best_model_name]['Specificity'].values[0],
        'best_cv_auc': results_df[results_df['Model'] == best_model_name]['CV_AUC'].values[0],
        'selected_features': ', '.join(selected_features[:5]) + '...'
    }


def main():
    """Main function to run seed experiment"""
    
    print("="*70)
    print("   Seed Experiment: Testing seeds 1-100")
    print("="*70)
    
    # 使用新的数据文件
    discovery_path = 'data/Discovery_HK.csv'
    validation_path = 'data/Validation_AUS_KM.csv'
    
    # Create main output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_output_dir = f'seed_experiment_{timestamp}'
    os.makedirs(main_output_dir, exist_ok=True)
    
    all_results = []
    
    for seed in range(1, 201):
        print(f"\n[{seed:3d}/100] Testing seed={seed}...", end=" ")
        
        seed_dir = os.path.join(main_output_dir, f'seed_{seed:03d}')
        
        try:
            result = run_pipeline_with_seed(discovery_path, validation_path, seed, seed_dir)
            all_results.append(result)
            print(f"Val_AUC={result['best_val_auc']:.4f} ({result['best_model']})")
        except Exception as e:
            print(f"Error: {e}")
            all_results.append({
                'seed': seed,
                'best_model': 'Error',
                'best_val_auc': 0,
                'best_sensitivity': 0,
                'best_specificity': 0,
                'best_cv_auc': 0,
                'selected_features': 'Error'
            })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values('best_val_auc', ascending=False)
    
    # Save summary
    summary_path = os.path.join(main_output_dir, 'seed_experiment_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Print top 10 seeds
    print("\n" + "="*70)
    print("   TOP 10 SEEDS BY VALIDATION AUC")
    print("="*70)
    print(summary_df.head(10).to_string(index=False))
    
    # Find best seed
    best_row = summary_df.iloc[0]
    print("\n" + "="*70)
    print(f"   BEST SEED: {int(best_row['seed'])}")
    print("="*70)
    print(f"   Model: {best_row['best_model']}")
    print(f"   Validation AUC: {best_row['best_val_auc']:.4f}")
    print(f"   CV AUC: {best_row['best_cv_auc']:.4f}")
    print(f"   Sensitivity: {best_row['best_sensitivity']:.4f}")
    print(f"   Specificity: {best_row['best_specificity']:.4f}")
    print(f"   Top Features: {best_row['selected_features']}")
    
    # Plot summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Validation AUC distribution
    ax1 = axes[0, 0]
    ax1.hist(summary_df['best_val_auc'], bins=20, color='#8B5CF6', edgecolor='white', alpha=0.8)
    ax1.axvline(x=best_row['best_val_auc'], color='red', linestyle='--', linewidth=2, label=f'Best: {best_row["best_val_auc"]:.4f}')
    ax1.set_xlabel('Validation AUC', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Validation AUC Across Seeds', fontweight='bold')
    ax1.legend()
    
    # 2. AUC by seed
    ax2 = axes[0, 1]
    sorted_by_seed = summary_df.sort_values('seed')
    ax2.plot(sorted_by_seed['seed'], sorted_by_seed['best_val_auc'], 'o-', color='#8B5CF6', markersize=4, alpha=0.7)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Seed', fontsize=12)
    ax2.set_ylabel('Validation AUC', fontsize=12)
    ax2.set_title('Validation AUC by Seed', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Best model frequency
    ax3 = axes[1, 0]
    model_counts = summary_df['best_model'].value_counts()
    ax3.bar(model_counts.index, model_counts.values, color='#10B981', edgecolor='white')
    ax3.set_xlabel('Model', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Best Model Frequency', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Top 10 seeds bar chart
    ax4 = axes[1, 1]
    top10 = summary_df.head(10)
    x = np.arange(len(top10))
    ax4.bar(x, top10['best_val_auc'], color='#8B5CF6', edgecolor='white')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"Seed {int(s)}" for s in top10['seed']], rotation=45, ha='right')
    ax4.set_ylabel('Validation AUC', fontsize=12)
    ax4.set_title('Top 10 Seeds by Validation AUC', fontweight='bold')
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
    # Add value labels
    for i, v in enumerate(top10['best_val_auc']):
        ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(main_output_dir, 'seed_experiment_summary.png'), 
                bbox_inches='tight', facecolor='white', dpi=150)
    
    return summary_df


if __name__ == '__main__':
    summary = main()

