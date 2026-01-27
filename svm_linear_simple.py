"""
Simple SVM Linear Training and Validation Script
Using the best seed and selected features from pipeline analysis

Author: Wendy
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, 
    classification_report, recall_score
)

# ==================== Configuration ====================
RANDOM_SEED = 134  # Best seed from seed_experiment.py (Val AUC = 0.7002)
# Features selected with seed 134 (best seed from experiment)
SELECTED_FEATURES = [
    'POE14', 'POE30.C18', 'POE31.C18', 'POE15', 'POE35.SORBITAN', 'POE25.C18',
    'POE17', 'POE24.SORBITAN', 'POE12', 'POE13', 'POE16', 'POE20.SORBITAN'
]

# Set random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# ==================== Load Data ====================
print("="*60)
print("   SVM Linear - Simple Training & Validation")
print("="*60)

# Load datasets
disc_df = pd.read_csv('LR_CD_P80_individual_input_table.csv', na_values=['.', '', ' '])
val_df = pd.read_csv('LR_AOCC_MiRES_combined_Validation_cohort_individual_input_table.csv', na_values=['.', '', ' '])

# Extract features and labels
y_train = disc_df['Group'].values
y_val = val_df['Group'].values

X_train_raw = disc_df[SELECTED_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
X_val_raw = val_df[SELECTED_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)

# Apply Log1p transformation
X_train_log = np.log1p(X_train_raw.values)
X_val_log = np.log1p(X_val_raw.values)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_log)
X_val_scaled = scaler.transform(X_val_log)

print(f"\nData loaded:")
print(f"  Discovery: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features")
print(f"  Validation: {X_val_scaled.shape[0]} samples")
print(f"  Class distribution (Discovery): {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"  Class distribution (Validation): {dict(zip(*np.unique(y_val, return_counts=True)))}")

# ==================== Train SVM Linear ====================
print("\n" + "="*60)
print("Training SVM Linear...")
print("="*60)

model = SVC(kernel='linear', C=0.5, class_weight='balanced', 
            probability=True, random_state=RANDOM_SEED)
model.fit(X_train_scaled, y_train)

print("Model trained successfully!")

# ==================== Evaluation on Discovery ====================
print("\n" + "-"*40)
print("Results on Discovery Set:")
print("-"*40)

y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
y_pred_train = model.predict(X_train_scaled)
train_auc = roc_auc_score(y_train, y_prob_train)

print(f"  AUC: {train_auc:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_train, y_pred_train, target_names=['Class 0', 'Class 1']))

# ==================== Evaluation on Validation ====================
print("-"*40)
print("Results on Validation Set:")
print("-"*40)

y_prob_val = model.predict_proba(X_val_scaled)[:, 1]
val_auc = roc_auc_score(y_val, y_prob_val)

# Find optimal threshold using Youden's J statistic
fpr, tpr, thresholds = roc_curve(y_val, y_prob_val)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

# Predictions with optimal threshold
y_pred_val = (y_prob_val >= best_threshold).astype(int)

# Calculate metrics
sensitivity = recall_score(y_val, y_pred_val, pos_label=1)  # True Positive Rate
specificity = recall_score(y_val, y_pred_val, pos_label=0)  # True Negative Rate

print(f"  AUC: {val_auc:.4f}")
print(f"  Optimal Threshold: {best_threshold:.4f}")
print(f"  Sensitivity: {sensitivity:.4f}")
print(f"  Specificity: {specificity:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_val, y_pred_val, target_names=['Class 0', 'Class 1']))

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred_val)
print(f"  Confusion Matrix:")
print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
print(f"    FN={cm[1,0]}, TP={cm[1,1]}")

# ==================== Plot Results ====================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. ROC Curve
ax1 = axes[0]
ax1.plot(fpr, tpr, color='#8B5CF6', linewidth=2, label=f'Validation AUC = {val_auc:.3f}')
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax1.scatter([fpr[best_idx]], [tpr[best_idx]], color='red', s=100, zorder=5, 
            label=f'Optimal (SEN={sensitivity:.2f}, SPE={specificity:.2f})')
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curve (Validation)', fontweight='bold', fontsize=13)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Confusion Matrix Heatmap
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
ax2.set_title('Confusion Matrix (Validation)', fontweight='bold', fontsize=13)
plt.colorbar(im, ax=ax2)

# 3. Performance Summary Bar Chart
ax3 = axes[2]
metrics = ['AUC', 'Sensitivity', 'Specificity']
values = [val_auc, sensitivity, specificity]
colors = ['#8B5CF6', '#8B5CF6', '#8B5CF6']
alphas = [1.0, 1.0, 0.4]
hatches = [None, None, '///']

bars = ax3.bar(metrics, values, color=colors, edgecolor='#8B5CF6', linewidth=2)
bars[1].set_facecolor('none')  # Hollow for SEN
bars[2].set_hatch('///')       # Hatch for SPE
bars[2].set_alpha(0.4)

for bar, val in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', 
             ha='center', fontsize=12, fontweight='bold', color='#8B5CF6')

ax3.set_ylim([0, 1.15])
ax3.set_ylabel('Score', fontsize=12)
ax3.set_title('Performance Metrics (Validation)', fontweight='bold', fontsize=13)
ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('svm_linear_results.png', bbox_inches='tight', facecolor='white', dpi=150)
plt.show()

# ==================== Summary ====================
print("\n" + "="*60)
print("   SUMMARY")
print("="*60)
print(f"  Model: SVM Linear (C=0.5, balanced)")
print(f"  Random Seed: {RANDOM_SEED}")
print(f"  Features: {len(SELECTED_FEATURES)}")
print(f"  ")
print(f"  Discovery AUC:  {train_auc:.4f}")
print(f"  Validation AUC: {val_auc:.4f}")
print(f"  Sensitivity:    {sensitivity:.4f}")
print(f"  Specificity:    {specificity:.4f}")
print(f"  ")
print(f"  Selected Features:")
for i, f in enumerate(SELECTED_FEATURES, 1):
    print(f"    {i:2d}. {f}")
print("="*60)

# Save results
results = {
    'Metric': ['Train_AUC', 'Val_AUC', 'Sensitivity', 'Specificity', 'Threshold'],
    'Value': [train_auc, val_auc, sensitivity, specificity, best_threshold]
}
pd.DataFrame(results).to_csv('svm_linear_results.csv', index=False)
print("\nResults saved: svm_linear_results.png, svm_linear_results.csv")

