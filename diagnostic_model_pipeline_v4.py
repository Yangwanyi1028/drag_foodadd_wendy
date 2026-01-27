"""
诊断模型Pipeline V4 - 数据变换优化版
重点改进：
1. 多种数据变换方法对比
2. 自动选择最佳变换
3. 改进的LASSO特征筛选

作者：Wendy
日期：2026-01-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, boxcox, yeojohnson
import warnings
import random
import os

warnings.filterwarnings('ignore')

# Set global random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

from sklearn.preprocessing import (
    StandardScaler, PowerTransformer, QuantileTransformer
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, confusion_matrix,
    recall_score, precision_score, f1_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except:
    HAS_XGBOOST = False

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

COLORS = {'primary': '#2E86AB', 'secondary': '#A23B72', 'accent': '#F18F01', 'success': '#C73E1D'}

# Unified style for AUC/SEN/SPE - All purple with different styles
METRIC_STYLES = {
    'AUC': {'color': '#8B5CF6', 'facecolor': '#8B5CF6', 'edgecolor': '#8B5CF6', 'hatch': None, 'alpha': 1.0},     # Purple solid
    'SEN': {'color': '#8B5CF6', 'facecolor': 'none', 'edgecolor': '#8B5CF6', 'hatch': None, 'alpha': 1.0},        # Purple hollow
    'SPE': {'color': '#8B5CF6', 'facecolor': '#8B5CF6', 'edgecolor': '#8B5CF6', 'hatch': '///', 'alpha': 0.4},    # Purple shadow/hatch
}


class DiagnosticModelPipelineV4:
    """带数据变换的诊断模型Pipeline"""
    
    def __init__(self, discovery_path, validation_path):
        # 读取数据
        disc_df = pd.read_csv(discovery_path, na_values=['.', '', ' '])
        val_df = pd.read_csv(validation_path, na_values=['.', '', ' '])
        
        self.y_discovery = disc_df['Group']
        self.y_validation = val_df['Group']
        
        X_disc = disc_df.drop('Group', axis=1)
        X_val = val_df.drop('Group', axis=1)
        
        # 共有特征
        common = list(set(X_disc.columns) & set(X_val.columns))
        self.X_discovery_raw = X_disc[common].apply(pd.to_numeric, errors='coerce').fillna(0)
        self.X_validation_raw = X_val[common].apply(pd.to_numeric, errors='coerce').fillna(0)
        self.feature_names = common
        
        print(f"Discovery: {self.X_discovery_raw.shape[0]} samples, {len(common)} features")
        print(f"Validation: {self.X_validation_raw.shape[0]} samples")
        print(f"Class distribution - Discovery: {dict(self.y_discovery.value_counts())}")
        print(f"Class distribution - Validation: {dict(self.y_validation.value_counts())}")
    
    # ==================== 1. 数据变换 ====================
    
    def compare_transformations(self):
        """比较不同数据变换方法的效果"""
        print("\n" + "="*60)
        print("Data Transformation Comparison")
        print("="*60)
        
        X_raw = self.X_discovery_raw.copy()
        y = self.y_discovery
        
        # 定义变换方法
        transformations = {
            'Raw': X_raw.values,
            'Log1p': np.log1p(X_raw.values),
            'Sqrt': np.sqrt(np.abs(X_raw.values)),
            'Yeo-Johnson': PowerTransformer(method='yeo-johnson').fit_transform(X_raw.values),
            'Quantile-Normal': QuantileTransformer(output_distribution='normal', random_state=42).fit_transform(X_raw.values),
        }
        
        results = {}
        
        for name, X_transformed in transformations.items():
            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_transformed)
            
            # 用简单模型评估
            lr = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc')
            
            # 计算特征的显著性提升
            n_significant = 0
            total_effect = 0
            for i, col in enumerate(self.feature_names):
                g0 = X_transformed[y == 0, i]
                g1 = X_transformed[y == 1, i]
                try:
                    _, p = mannwhitneyu(g0, g1)
                    if p < 0.05:
                        n_significant += 1
                    # 效应量
                    U, _ = mannwhitneyu(g0, g1)
                    r = abs(1 - (2*U)/(len(g0)*len(g1)))
                    total_effect += r
                except:
                    pass
            
            results[name] = {
                'cv_auc': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'n_significant': n_significant,
                'avg_effect': total_effect / len(self.feature_names)
            }
            
            print(f"\n{name}:")
            print(f"  CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
            print(f"  Significant features (p<0.05): {n_significant}")
            print(f"  Average effect size: {total_effect / len(self.feature_names):.4f}")
        
        # 选择最佳变换
        best_transform = max(results.keys(), key=lambda k: results[k]['cv_auc'])
        self.best_transform = best_transform
        print(f"\nBest transformation: {best_transform}")
        
        # 可视化
        self._plot_transformation_comparison(transformations, results)
        
        self.transform_results = results
        return results
    
    def _plot_transformation_comparison(self, transformations, results):
        """可视化变换对比"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. AUC对比
        ax1 = axes[0, 0]
        names = list(results.keys())
        aucs = [results[n]['cv_auc'] for n in names]
        stds = [results[n]['cv_std'] for n in names]
        colors = [COLORS['success'] if n == self.best_transform else COLORS['primary'] for n in names]
        
        bars = ax1.bar(range(len(names)), aucs, yerr=stds, capsize=5, color=colors)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=30, ha='right')
        ax1.set_ylabel('CV AUC')
        ax1.set_title('CV AUC Comparison by Transformation', fontweight='bold')
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax1.set_ylim([0.45, 0.75])
        
        # Add value annotations
        for bar, auc in zip(bars, aucs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{auc:.3f}', ha='center', fontsize=9)
        
        # 2. Significant features comparison
        ax2 = axes[0, 1]
        n_sigs = [results[n]['n_significant'] for n in names]
        ax2.bar(range(len(names)), n_sigs, color=colors)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=30, ha='right')
        ax2.set_ylabel('Significant Features (p<0.05)')
        ax2.set_title('Significant Features Count', fontweight='bold')
        
        # 3. Average effect size comparison
        ax3 = axes[0, 2]
        effects = [results[n]['avg_effect'] for n in names]
        ax3.bar(range(len(names)), effects, color=colors)
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=30, ha='right')
        ax3.set_ylabel('Average Effect Size')
        ax3.set_title('Average Effect Size Comparison', fontweight='bold')
        
        # 4-6. 选一个特征展示变换效果
        # 找一个显著特征
        best_feat_idx = 0
        best_p = 1.0
        for i, col in enumerate(self.feature_names):
            g0 = self.X_discovery_raw[self.y_discovery == 0].iloc[:, i]
            g1 = self.X_discovery_raw[self.y_discovery == 1].iloc[:, i]
            try:
                _, p = mannwhitneyu(g0, g1)
                if p < best_p:
                    best_p = p
                    best_feat_idx = i
            except:
                pass
        
        feat_name = self.feature_names[best_feat_idx]
        
        # Raw分布
        ax4 = axes[1, 0]
        g0_raw = self.X_discovery_raw[self.y_discovery == 0].iloc[:, best_feat_idx]
        g1_raw = self.X_discovery_raw[self.y_discovery == 1].iloc[:, best_feat_idx]
        ax4.hist(g0_raw, bins=20, alpha=0.6, label='Group 0', color=COLORS['primary'])
        ax4.hist(g1_raw, bins=20, alpha=0.6, label='Group 1', color=COLORS['secondary'])
        ax4.set_xlabel(f'{feat_name} (Raw)')
        ax4.set_title('Raw Data Distribution', fontweight='bold')
        ax4.legend()
        
        # Log transformation
        ax5 = axes[1, 1]
        g0_log = np.log1p(g0_raw)
        g1_log = np.log1p(g1_raw)
        ax5.hist(g0_log, bins=20, alpha=0.6, label='Group 0', color=COLORS['primary'])
        ax5.hist(g1_log, bins=20, alpha=0.6, label='Group 1', color=COLORS['secondary'])
        ax5.set_xlabel(f'{feat_name} (Log1p)')
        ax5.set_title('Log Transformed Distribution', fontweight='bold')
        ax5.legend()
        
        # Quantile transformation
        ax6 = axes[1, 2]
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        X_qt = qt.fit_transform(self.X_discovery_raw.values)
        g0_qt = X_qt[self.y_discovery == 0, best_feat_idx]
        g1_qt = X_qt[self.y_discovery == 1, best_feat_idx]
        ax6.hist(g0_qt, bins=20, alpha=0.6, label='Group 0', color=COLORS['primary'])
        ax6.hist(g1_qt, bins=20, alpha=0.6, label='Group 1', color=COLORS['secondary'])
        ax6.set_xlabel(f'{feat_name} (Quantile-Normal)')
        ax6.set_title('Quantile Transformed Distribution', fontweight='bold')
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig('v4_01_transformation_comparison.png', bbox_inches='tight', facecolor='white')
        plt.show()
    
    # ==================== 2. 应用最佳变换 ====================
    
    def apply_best_transformation(self, method='Log1p'):
        """Apply specified data transformation"""
        print("\n" + "="*60)
        print(f"Applying transformation: {method}")
        print("="*60)
        
        if method == 'Raw':
            self.X_discovery = self.X_discovery_raw.copy()
            self.X_validation = self.X_validation_raw.copy()
            self.transformer = None
        elif method == 'Log1p':
            self.X_discovery = pd.DataFrame(
                np.log1p(self.X_discovery_raw.values),
                columns=self.feature_names,
                index=self.X_discovery_raw.index
            )
            self.X_validation = pd.DataFrame(
                np.log1p(self.X_validation_raw.values),
                columns=self.feature_names,
                index=self.X_validation_raw.index
            )
            self.transformer = 'log1p'
        elif method == 'Sqrt':
            self.X_discovery = pd.DataFrame(
                np.sqrt(np.abs(self.X_discovery_raw.values)),
                columns=self.feature_names,
                index=self.X_discovery_raw.index
            )
            self.X_validation = pd.DataFrame(
                np.sqrt(np.abs(self.X_validation_raw.values)),
                columns=self.feature_names,
                index=self.X_validation_raw.index
            )
            self.transformer = 'sqrt'
        elif method == 'Yeo-Johnson':
            self.transformer = PowerTransformer(method='yeo-johnson')
            self.X_discovery = pd.DataFrame(
                self.transformer.fit_transform(self.X_discovery_raw.values),
                columns=self.feature_names,
                index=self.X_discovery_raw.index
            )
            self.X_validation = pd.DataFrame(
                self.transformer.transform(self.X_validation_raw.values),
                columns=self.feature_names,
                index=self.X_validation_raw.index
            )
        elif method == 'Quantile-Normal':
            self.transformer = QuantileTransformer(output_distribution='normal', random_state=42)
            self.X_discovery = pd.DataFrame(
                self.transformer.fit_transform(self.X_discovery_raw.values),
                columns=self.feature_names,
                index=self.X_discovery_raw.index
            )
            self.X_validation = pd.DataFrame(
                self.transformer.transform(self.X_validation_raw.values),
                columns=self.feature_names,
                index=self.X_validation_raw.index
            )
        
        print(f"Transformation completed")
    
    # ==================== 3. 特征筛选 ====================
    
    def feature_selection(self, n_features=15):
        """Feature selection"""
        print("\n" + "="*60)
        print("Feature Selection")
        print("="*60)
        
        X = self.X_discovery
        y = self.y_discovery
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        feature_scores = pd.DataFrame(index=self.feature_names)
        
        # 1. Mann-Whitney U
        print("1. Mann-Whitney U test...")
        mw_scores = {}
        for col in self.feature_names:
            g0 = X[y == 0][col]
            g1 = X[y == 1][col]
            try:
                _, p = mannwhitneyu(g0, g1)
                mw_scores[col] = -np.log10(p + 1e-300)
            except:
                mw_scores[col] = 0
        feature_scores['MW_score'] = pd.Series(mw_scores)
        feature_scores['MW_rank'] = feature_scores['MW_score'].rank(ascending=False)
        
        # 2. Effect size
        print("2. Effect size...")
        effect_scores = {}
        for col in self.feature_names:
            g0 = X[y == 0][col]
            g1 = X[y == 1][col]
            try:
                U, _ = mannwhitneyu(g0, g1)
                r = abs(1 - (2*U)/(len(g0)*len(g1)))
                effect_scores[col] = r
            except:
                effect_scores[col] = 0
        feature_scores['Effect'] = pd.Series(effect_scores)
        feature_scores['Effect_rank'] = feature_scores['Effect'].rank(ascending=False)
        
        # 3. L1 Logistic (LASSO alternative)
        print("3. L1 regularization...")
        best_coefs = np.zeros(len(self.feature_names))
        for C in [0.01, 0.05, 0.1, 0.5, 1.0]:
            lr_l1 = LogisticRegression(penalty='l1', solver='saga', C=C, 
                                       class_weight='balanced', max_iter=5000, random_state=42)
            lr_l1.fit(X_scaled, y)
            coefs = np.abs(lr_l1.coef_[0])
            n_nonzero = np.sum(coefs > 0)
            if 10 <= n_nonzero <= 40:
                best_coefs = coefs
                print(f"    C={C}: {n_nonzero} non-zero features")
                break
            elif n_nonzero > 0:
                best_coefs = coefs
        
        feature_scores['L1_coef'] = best_coefs
        feature_scores['L1_rank'] = feature_scores['L1_coef'].rank(ascending=False, method='min')
        
        # 4. Random Forest
        print("4. Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=3, 
                                   class_weight='balanced', random_state=42)
        rf.fit(X_scaled, y)
        feature_scores['RF'] = rf.feature_importances_
        feature_scores['RF_rank'] = feature_scores['RF'].rank(ascending=False)
        
        # 5. Mutual Information
        print("5. Mutual Information...")
        mi = mutual_info_classif(X_scaled, y, random_state=42)
        feature_scores['MI'] = mi
        feature_scores['MI_rank'] = feature_scores['MI'].rank(ascending=False)
        
        # Combined ranking (higher weight for statistical methods)
        feature_scores['Avg_rank'] = (
            feature_scores['MW_rank'] * 2 +
            feature_scores['Effect_rank'] * 2 +
            feature_scores['L1_rank'] * 1.5 +
            feature_scores['RF_rank'] * 1 +
            feature_scores['MI_rank'] * 1
        ) / 7.5
        
        feature_scores = feature_scores.sort_values('Avg_rank')
        
        # Select Top features
        selected = feature_scores.head(n_features).index.tolist()
        
        print(f"\nSelected {len(selected)} features:")
        for i, f in enumerate(selected[:10], 1):
            s = feature_scores.loc[f]
            print(f"  {i}. {f}: MW={s['MW_score']:.2f}, Effect={s['Effect']:.3f}, L1={s['L1_coef']:.3f}")
        
        self.feature_scores = feature_scores
        self.selected_features = selected
        
        # 保存
        feature_scores.to_csv('v4_feature_ranking.csv')
        
        return selected
    
    # ==================== 4. 模型训练 ====================
    
    def train_models(self, cv_folds=5):
        """Train models"""
        print("\n" + "="*60)
        print("Model Training")
        print("="*60)
        
        X = self.X_discovery[self.selected_features]
        y = self.y_discovery
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        X_val = self.X_validation[self.selected_features]
        X_val_scaled = self.scaler.transform(X_val)
        y_val = self.y_validation
        
        models = {
            'Logistic (L2)': LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42),
            'Logistic (L1)': LogisticRegression(C=0.5, penalty='l1', solver='saga', 
                                                class_weight='balanced', max_iter=1000, random_state=42),
            'SVM Linear': SVC(kernel='linear', C=0.5, class_weight='balanced', probability=True, random_state=42),
            'SVM RBF': SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=3, 
                                                    class_weight='balanced', random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=2, 
                                                           learning_rate=0.1, random_state=42),
        }
        
        if HAS_XGBOOST:
            scale_weight = sum(y == 0) / sum(y == 1)
            models['XGBoost'] = XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1,
                                              scale_pos_weight=scale_weight, reg_alpha=0.5,
                                              random_state=42, use_label_encoder=False, eval_metric='logloss')
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
            model.fit(X_scaled, y)
            
            y_prob_train = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_scaled)
            y_prob_val = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_val_scaled)
            
            train_auc = roc_auc_score(y, y_prob_train)
            val_auc = roc_auc_score(y_val, y_prob_val)
            
            fpr, tpr, _ = roc_curve(y_val, y_prob_val)
            
            results[name] = {
                'model': model,
                'cv_auc': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_auc': train_auc,
                'val_auc': val_auc,
                'fpr': fpr,
                'tpr': tpr,
                'y_prob_val': y_prob_val
            }
            
            gap = train_auc - val_auc
            status = "OK" if gap < 0.1 else "Overfitting"
            print(f"  CV: {cv_scores.mean():.4f}+/-{cv_scores.std():.4f}, Val: {val_auc:.4f} [{status}]")
        
        self.model_results = results
        
        # 选最佳模型
        best_name = max(results.keys(), key=lambda k: results[k]['val_auc'])
        self.best_model_name = best_name
        self.best_model = results[best_name]['model']
        
        print(f"\nBest model: {best_name}, Validation AUC: {results[best_name]['val_auc']:.4f}")
        
        return results
    
    # ==================== 5. 可视化 ====================
    
    def plot_results(self):
        """Plot results"""
        print("\n" + "="*60)
        print("Results Visualization")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Feature ranking
        ax1 = axes[0, 0]
        top_features = self.feature_scores.head(15)
        ax1.barh(range(len(top_features)), top_features['Avg_rank'].values, color=COLORS['primary'])
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features.index, fontsize=8)
        ax1.set_xlabel('Combined Rank Score (lower is better)')
        ax1.set_title('Top 15 Feature Ranking', fontweight='bold')
        ax1.invert_yaxis()
        
        # 2. Method ranking heatmap
        ax2 = axes[0, 1]
        rank_cols = ['MW_rank', 'Effect_rank', 'L1_rank', 'RF_rank', 'MI_rank']
        rank_data = top_features[rank_cols].copy()
        rank_data.columns = ['Mann-Whitney', 'Effect Size', 'L1', 'RF', 'MI']
        sns.heatmap(rank_data, ax=ax2, cmap='RdYlGn_r', annot=True, fmt='.0f', linewidths=0.5)
        ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=8)
        ax2.set_title('Ranking by Different Methods', fontweight='bold')
        
        # 3. Validation ROC
        ax3 = axes[0, 2]
        colors_list = plt.cm.Set2(np.linspace(0, 1, len(self.model_results)))
        
        for (name, result), color in zip(self.model_results.items(), colors_list):
            ax3.plot(result['fpr'], result['tpr'],
                    label=f"{name} ({result['val_auc']:.3f})",
                    linewidth=2 if name == self.best_model_name else 1.5,
                    color=color,
                    linestyle='-' if name == self.best_model_name else '--')
        
        ax3.plot([0, 1], [0, 1], 'k:', linewidth=1)
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('Validation ROC Curves', fontweight='bold')
        ax3.legend(loc='lower right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. CV vs Validation AUC
        ax4 = axes[1, 0]
        names = list(self.model_results.keys())
        cv_aucs = [self.model_results[n]['cv_auc'] for n in names]
        val_aucs = [self.model_results[n]['val_auc'] for n in names]
        
        x = np.arange(len(names))
        width = 0.35
        ax4.bar(x - width/2, cv_aucs, width, label='CV AUC', color=COLORS['primary'])
        ax4.bar(x + width/2, val_aucs, width, label='Validation AUC', color=COLORS['secondary'])
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax4.set_xticks(x)
        ax4.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('AUC')
        ax4.set_title('CV vs Validation AUC', fontweight='bold')
        ax4.legend()
        ax4.set_ylim([0.4, 0.8])
        
        # 5. Confusion matrix
        ax5 = axes[1, 1]
        y_prob = self.model_results[self.best_model_name]['y_prob_val']
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(self.y_validation, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                   xticklabels=['Pred 0', 'Pred 1'],
                   yticklabels=['True 0', 'True 1'])
        ax5.set_title(f'{self.best_model_name} Confusion Matrix', fontweight='bold')
        
        # 6. Summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        best_result = self.model_results[self.best_model_name]
        
        summary = f"""
    ============================================
              V4 Pipeline Summary
    ============================================
      Transformation: {self.best_transform:<20}
      
      Discovery samples:  {self.X_discovery.shape[0]:>6}
      Validation samples: {self.X_validation.shape[0]:>6}
      Selected features:  {len(self.selected_features):>6}
    --------------------------------------------
      Best Model: {self.best_model_name:<20}
      CV AUC: {best_result['cv_auc']:.4f} +/- {best_result['cv_std']:.4f}
      Validation AUC: {best_result['val_auc']:.4f}
      Generalization Gap: {best_result['train_auc'] - best_result['val_auc']:.4f}
    --------------------------------------------
      Top 10 Features:"""
        
        for i, f in enumerate(self.selected_features[:10], 1):
            summary += f"\n        {i:2}. {f}"
        
        summary += """
    ============================================"""
        
        ax6.text(0.02, 0.98, summary, transform=ax6.transAxes,
                fontsize=9, fontfamily='monospace', verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig('v4_02_final_results.png', bbox_inches='tight', facecolor='white', dpi=150)
        plt.show()
        
        # 保存模型对比
        comparison_data = []
        for name, r in self.model_results.items():
            comparison_data.append({
                'Model': name,
                'CV_AUC': r['cv_auc'],
                'CV_std': r['cv_std'],
                'Train_AUC': r['train_auc'],
                'Val_AUC': r['val_auc'],
                'Gap': r['train_auc'] - r['val_auc']
            })
        pd.DataFrame(comparison_data).sort_values('Val_AUC', ascending=False).to_csv(
            'v4_model_comparison.csv', index=False)
        
        pd.DataFrame({'Features': self.selected_features}).to_csv(
            'v4_selected_features.csv', index=False)
        
        print("\nResults saved: v4_feature_ranking.csv, v4_model_comparison.csv, v4_selected_features.csv")
    
    # ==================== 辅助方法 ====================
    
    def _get_model_by_name(self, model_name):
        """根据名称返回模型实例"""
        if model_name == 'Logistic (L2)':
            return LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
        elif model_name == 'Logistic (L1)':
            return LogisticRegression(C=0.5, penalty='l1', solver='saga', 
                                     class_weight='balanced', max_iter=1000, random_state=42)
        elif model_name == 'SVM Linear':
            return SVC(kernel='linear', C=0.5, class_weight='balanced', probability=True, random_state=42)
        elif model_name == 'SVM RBF':
            return SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=42)
        elif model_name == 'Random Forest':
            return RandomForestClassifier(n_estimators=100, max_depth=3, 
                                         class_weight='balanced', random_state=42)
        elif model_name == 'Gradient Boosting':
            return GradientBoostingClassifier(n_estimators=50, max_depth=2, 
                                             learning_rate=0.1, random_state=42)
        elif model_name == 'XGBoost' and HAS_XGBOOST:
            return XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1,
                                scale_pos_weight=4.35, reg_alpha=0.5,
                                random_state=42, use_label_encoder=False, eval_metric='logloss')
        else:
            # 默认返回SVM Linear（通常泛化性能最好）
            return SVC(kernel='linear', C=0.5, class_weight='balanced', probability=True, random_state=42)
    
    # ==================== 6. 特征数量递增实验 ====================
    
    def feature_incremental_experiment(self, feature_range=range(3, 16), model_name=None):
        """
        用不同数量的Top特征训练模型，对比性能
        
        Args:
            feature_range: 特征数量范围，如 range(3, 16) 表示 top3 到 top15
            model_name: 使用的模型名称，如果为None则自动选择验证集上最好的模型
        """
        print("\n" + "="*60)
        print("Feature Incremental Experiment")
        print("="*60)
        
        # Ensure feature ranking is complete
        if not hasattr(self, 'feature_scores'):
            raise ValueError("Please run feature_selection() first")
        
        # Auto-select best model if not specified
        if model_name is None:
            if hasattr(self, 'best_model_name'):
                model_name = self.best_model_name
                print(f"Auto-selected best model: {model_name}")
            else:
                model_name = 'SVM Linear'
                print(f"Using default model: {model_name}")
        
        # 获取排序后的特征列表
        sorted_features = self.feature_scores.sort_values('Avg_rank').index.tolist()
        
        results = []
        
        for n_feat in feature_range:
            # 选择Top n特征
            selected = sorted_features[:n_feat]
            
            X = self.X_discovery[selected]
            y = self.y_discovery
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_val = self.X_validation[selected]
            X_val_scaled = scaler.transform(X_val)
            y_val = self.y_validation
            
            # 定义模型
            model = self._get_model_by_name(model_name)
            
            # CV
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
            
            # 训练
            model.fit(X_scaled, y)
            
            # 验证集预测
            y_prob = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_scaled)
            y_prob_val = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_val_scaled)
            
            train_auc = roc_auc_score(y, y_prob)
            val_auc = roc_auc_score(y_val, y_prob_val)
            
            # 找最优阈值（基于Youden's J）
            fpr, tpr, thresholds = roc_curve(y_val, y_prob_val)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            best_thresh = thresholds[best_idx]
            
            # 用最优阈值计算指标
            y_pred = (y_prob_val >= best_thresh).astype(int)
            
            # 敏感性 = 召回率 (对于正类)
            sensitivity = recall_score(y_val, y_pred, pos_label=1)
            # 特异性 = 召回率 (对于负类)
            specificity = recall_score(y_val, y_pred, pos_label=0)
            
            results.append({
                'n_features': n_feat,
                'features': selected,
                'cv_auc': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_auc': train_auc,
                'val_auc': val_auc,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'threshold': best_thresh
            })
            
            print(f"Top {n_feat:2d}: CV={cv_scores.mean():.3f}, Val_AUC={val_auc:.3f}, "
                  f"SEN={sensitivity:.3f}, SPE={specificity:.3f}")
        
        self.incremental_results = results
        
        # Save results to CSV (no plotting for v4_03)
        results_df = pd.DataFrame([{
            'n_features': r['n_features'],
            'CV_AUC': r['cv_auc'],
            'CV_std': r['cv_std'],
            'Train_AUC': r['train_auc'],
            'Val_AUC': r['val_auc'],
            'Sensitivity': r['sensitivity'],
            'Specificity': r['specificity'],
            'Threshold': r['threshold'],
            'Features': ', '.join(r['features'])
        } for r in results])
        results_df.to_csv('v4_feature_incremental_results.csv', index=False)
        
        # Find best config
        best_val_idx = np.argmax([r['val_auc'] for r in results])
        best_result = results[best_val_idx]
        print(f"\nBest config: Top {best_result['n_features']} features")
        print(f"   Validation AUC: {best_result['val_auc']:.4f}")
        print(f"   Sensitivity: {best_result['sensitivity']:.4f}")
        print(f"   Specificity: {best_result['specificity']:.4f}")
        print(f"   Features: {', '.join(best_result['features'][:5])}...")
        print("\nResults saved: v4_feature_incremental_results.csv")
        
        return results
    
    def run_multiple_models_incremental(self, feature_range=range(3, 13)):
        """Run feature incremental experiment for multiple models"""
        print("\n" + "="*60)
        print("Multi-Model Feature Incremental Experiment")
        print("="*60)
        
        # All models to test
        models_to_test = [
            'Logistic (L2)', 'Logistic (L1)', 
            'SVM Linear', 'SVM RBF',
            'Random Forest', 'Gradient Boosting'
        ]
        if HAS_XGBOOST:
            models_to_test.append('XGBoost')
        
        all_results = {}
        
        for model_name in models_to_test:
            print(f"\n{'='*40}")
            print(f"Model: {model_name}")
            print('='*40)
            
            sorted_features = self.feature_scores.sort_values('Avg_rank').index.tolist()
            results = []
            
            for n_feat in feature_range:
                selected = sorted_features[:n_feat]
                
                X = self.X_discovery[selected]
                y = self.y_discovery
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                X_val = self.X_validation[selected]
                X_val_scaled = scaler.transform(X_val)
                y_val = self.y_validation
                
                model = self._get_model_by_name(model_name)
                
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
                
                model.fit(X_scaled, y)
                
                # Get probabilities
                if hasattr(model, 'predict_proba'):
                    y_prob_val = model.predict_proba(X_val_scaled)[:, 1]
                else:
                    y_prob_val = model.decision_function(X_val_scaled)
                
                val_auc = roc_auc_score(y_val, y_prob_val)
                
                fpr, tpr, thresholds = roc_curve(y_val, y_prob_val)
                best_idx = np.argmax(tpr - fpr)
                best_thresh = thresholds[best_idx]
                
                y_pred = (y_prob_val >= best_thresh).astype(int)
                sensitivity = recall_score(y_val, y_pred, pos_label=1)
                specificity = recall_score(y_val, y_pred, pos_label=0)
                
                results.append({
                    'n_features': n_feat,
                    'cv_auc': cv_scores.mean(),
                    'val_auc': val_auc,
                    'sensitivity': sensitivity,
                    'specificity': specificity
                })
            
            all_results[model_name] = results
            
            # Print best result for this model
            best_idx = np.argmax([r['val_auc'] for r in results])
            best = results[best_idx]
            print(f"Best: Top{best['n_features']}, Val_AUC={best['val_auc']:.3f}, "
                  f"SEN={best['sensitivity']:.3f}, SPE={best['specificity']:.3f}")
        
        # Plot multi-model comparison
        self._plot_multi_model_comparison(all_results, feature_range)
        
        return all_results
    
    def _plot_multi_model_comparison(self, all_results, feature_range):
        """Plot multi-model comparison - only bar chart"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Find best configuration for each model
        best_configs = []
        for model_name, results in all_results.items():
            best_idx = np.argmax([r['val_auc'] for r in results])
            best = results[best_idx]
            best_configs.append({
                'model': model_name,
                'n_features': best['n_features'],
                'val_auc': best['val_auc'],
                'sensitivity': best['sensitivity'],
                'specificity': best['specificity']
            })
        
        # Sort by validation AUC
        best_configs = sorted(best_configs, key=lambda x: x['val_auc'], reverse=True)
        
        x = np.arange(len(best_configs))
        width = 0.25
        
        # AUC: Purple solid
        ax.bar(x - width, [c['val_auc'] for c in best_configs], width, 
               label='AUC', color=METRIC_STYLES['AUC']['facecolor'], 
               edgecolor=METRIC_STYLES['AUC']['edgecolor'], linewidth=1.5)
        # SEN: Purple hollow (only edge)
        ax.bar(x, [c['sensitivity'] for c in best_configs], width, 
               label='Sensitivity', facecolor='none', 
               edgecolor=METRIC_STYLES['SEN']['edgecolor'], linewidth=2)
        # SPE: Purple shadow (hatched with alpha)
        ax.bar(x + width, [c['specificity'] for c in best_configs], width, 
               label='Specificity', color=METRIC_STYLES['SPE']['facecolor'], 
               edgecolor=METRIC_STYLES['SPE']['edgecolor'],
               alpha=METRIC_STYLES['SPE']['alpha'], 
               hatch=METRIC_STYLES['SPE']['hatch'], linewidth=1)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c['model']}\n(Top{c['n_features']})" for c in best_configs], 
                           fontsize=10, rotation=15, ha='right')
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title('Best Config AUC/SEN/SPE by Model (Sorted by AUC)', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11, loc='upper right')
        ax.set_ylim([0, 1.15])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value annotations
        for i, c in enumerate(best_configs):
            ax.text(i - width, c['val_auc'] + 0.02, f"{c['val_auc']:.2f}", ha='center', fontsize=9, 
                    color=METRIC_STYLES['AUC']['color'], fontweight='bold')
            ax.text(i, c['sensitivity'] + 0.02, f"{c['sensitivity']:.2f}", ha='center', fontsize=9,
                    color=METRIC_STYLES['SEN']['color'], fontweight='bold')
            ax.text(i + width, c['specificity'] + 0.02, f"{c['specificity']:.2f}", ha='center', fontsize=9,
                    color=METRIC_STYLES['SPE']['color'], fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('v4_04_multi_model_comparison.png', bbox_inches='tight', facecolor='white', dpi=150)
        plt.show()
        
        # Save detailed results to CSV
        results_df = pd.DataFrame(best_configs)
        results_df.to_csv('v4_multi_model_best_configs.csv', index=False)
        
        print("\nResults saved: v4_04_multi_model_comparison.png, v4_multi_model_best_configs.csv")
    
    # ==================== 主运行 ====================
    
    def run(self, n_features=12, transform_method=None, run_incremental=True):
        """Run complete pipeline"""
        print("\n" + "="*70)
        print("   Diagnostic Model Pipeline V4 - Transformation Optimized")
        print("="*70)
        
        # 1. 比较变换方法
        self.compare_transformations()
        
        # 2. 应用最佳变换（或指定变换）
        method = transform_method if transform_method else self.best_transform
        self.apply_best_transformation(method)
        
        # 3. 特征筛选
        self.feature_selection(n_features=n_features)
        
        # 4. 模型训练
        self.train_models()
        
        # 5. 可视化
        self.plot_results()
        
        # 6. 特征数量递增实验
        if run_incremental:
            self.feature_incremental_experiment(feature_range=range(3, n_features+1))
            self.run_multiple_models_incremental(feature_range=range(3, n_features+1))
        
        print("\n" + "="*70)
        print("   Pipeline V4 Complete!")
        print("="*70)


# ==================== 主程序 ====================

if __name__ == '__main__':
    pipeline = DiagnosticModelPipelineV4(
        'LR_CD_P80_individual_input_table.csv',
        'LR_AOCC_MiRES_combined_Validation_cohort_individual_input_table.csv'
    )
    
    # Run with specified transformation, or let the program auto-select
    # transform_method options: 'Raw', 'Log1p', 'Sqrt', 'Yeo-Johnson', 'Quantile-Normal'
    pipeline.run(n_features=12, transform_method=None)  # None means auto-select best

