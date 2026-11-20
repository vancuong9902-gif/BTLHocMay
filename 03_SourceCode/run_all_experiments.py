import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve, validation_curve, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, recall_score, f1_score, precision_score,
                             accuracy_score, confusion_matrix, roc_curve)

sys.path.append(os.path.abspath('..'))
from src.features import load_and_split_data, get_preprocessor_pipeline

base_dir = str(Path(__file__).resolve().parent)
data_path = os.path.join(base_dir, 'data', 'framingham.csv')
X_train, X_test, y_train, y_test = load_and_split_data(data_path=data_path, test_size=0.2, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_dir = os.path.join(base_dir, 'models', 'results')
figures_dir = os.path.join(base_dir, 'notebooks', 'figures')
os.makedirs(figures_dir, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================
# 1. RECREATE AND SAVE ALL 4 MODELS
# ============================================================

print("=" * 60)
print("RECREATING ALL 4 MODELS")
print("=" * 60)

# 1.1 Baseline Logistic Regression
print("\n1. Baseline Logistic Regression...")
lr_pipeline = Pipeline(steps=[
    ('preprocessor', get_preprocessor_pipeline()),
    ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
])
lr_pipeline.fit(X_train, y_train)
joblib.dump(lr_pipeline, os.path.join(base_dir, 'models', 'best_lr.joblib'))

# 1.2 k-NN (from previous GridSearchCV)
print("2. k-NN...")
knn_results = pd.read_csv(os.path.join(results_dir, 'knn_gridsearch_results.csv'))
best_knn_idx = knn_results['rank_test_score'].idxmin()
best_knn_params = {
    'n_neighbors': int(knn_results.loc[best_knn_idx, 'param_classifier__n_neighbors']),
    'weights': knn_results.loc[best_knn_idx, 'param_classifier__weights'],
    'p': int(knn_results.loc[best_knn_idx, 'param_classifier__p']),
    'metric': 'minkowski'
}
knn_pipeline = Pipeline(steps=[
    ('preprocessor', get_preprocessor_pipeline()),
    ('classifier', KNeighborsClassifier(**best_knn_params))
])
knn_pipeline.fit(X_train, y_train)
joblib.dump(knn_pipeline, os.path.join(base_dir, 'models', 'best_knn.joblib'))
print(f"   Best k-NN params: {best_knn_params}")

# 1.3 Linear SVM (already saved)
print("3. Linear SVM (loaded)...")
best_svm_linear = joblib.load(os.path.join(base_dir, 'models', 'best_svm_linear.joblib'))

# 1.4 RBF SVM (already saved)
print("4. RBF SVM (loaded)...")
best_svm_rbf = joblib.load(os.path.join(base_dir, 'models', 'best_svm_rbf.joblib'))

# ============================================================
# 2. CROSS-VALIDATION FOR ALL MODELS (5-fold, all metrics)
# ============================================================

print("\n" + "=" * 60)
print("CROSS-VALIDATION (5-Fold Stratified)")
print("=" * 60)

models_cv = {}
for name, pipeline in [
    ('Logistic Regression', lr_pipeline),
    ('k-NN', knn_pipeline),
    ('Linear SVM', best_svm_linear),
    ('RBF SVM', best_svm_rbf)
]:
    print(f"\nRunning CV for {name}...")
    start = time.time()
    cv_res = cross_validate(
        pipeline, X_train, y_train, cv=cv,
        scoring=['roc_auc', 'recall', 'f1', 'precision', 'accuracy'],
        n_jobs=-1
    )
    models_cv[name] = cv_res
    print(f"  Done in {time.time()-start:.1f}s")
    for metric in ['roc_auc', 'recall', 'f1', 'precision', 'accuracy']:
        scores = cv_res[f'test_{metric}']
        print(f"  {metric.upper():10s}: {scores.mean():.4f} ± {scores.std():.4f}")

# ============================================================
# 3. TEST SET EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("HOLDOUT TEST SET EVALUATION")
print("=" * 60)

test_metrics = {}
for name, pipeline in [
    ('Logistic Regression', lr_pipeline),
    ('k-NN', knn_pipeline),
    ('Linear SVM', best_svm_linear),
    ('RBF SVM', best_svm_rbf)
]:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Accuracy': accuracy_score(y_test, y_pred)
    }
    test_metrics[name] = metrics
    print(f"\n{name}:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")

# ============================================================
# 4. SAVE COMPARISON TABLES
# ============================================================

print("\n" + "=" * 60)
print("SAVING COMPARISON TABLES")
print("=" * 60)

# 4.1 CV Comparison
cv_rows = []
for name, res in models_cv.items():
    row = {'Model': name}
    for metric in ['roc_auc', 'recall', 'f1', 'precision', 'accuracy']:
        scores = res[f'test_{metric}']
        row[f'CV {metric.upper()} Mean'] = f'{scores.mean():.4f}'
        row[f'CV {metric.upper()} Std'] = f'{scores.std():.4f}'
    cv_rows.append(row)
cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(os.path.join(results_dir, 'cv_comparison.csv'), index=False)
print("Saved: cv_comparison.csv")

# 4.2 Test Comparison
test_rows = []
for name, metrics in test_metrics.items():
    row = {'Model': name}
    row.update(metrics)
    test_rows.append(row)
test_df = pd.DataFrame(test_rows)
test_df.to_csv(os.path.join(results_dir, 'test_comparison.csv'), index=False)
print("Saved: test_comparison.csv")

# 4.3 Full Comparison
full_rows = []
for name in ['Logistic Regression', 'k-NN', 'Linear SVM', 'RBF SVM']:
    cv_res = models_cv[name]
    test_res = test_metrics[name]
    row = {'Model': name}
    for metric in ['roc_auc', 'recall', 'f1', 'precision', 'accuracy']:
        scores = cv_res[f'test_{metric}']
        row[f'CV {metric.upper()} (Mean±Std)'] = f'{scores.mean():.4f}±{scores.std():.4f}'
        row[f'Test {metric.upper()}'] = f'{test_res[metric.upper()]:.4f}'
    full_rows.append(row)
full_df = pd.DataFrame(full_rows)
full_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
print("Saved: model_comparison.csv")

print("\nFull comparison:")
print(full_df.to_string(index=False))

# ============================================================
# 5. CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 60)
print("GENERATING CONFUSION MATRICES")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
for idx, (name, pipeline) in enumerate([
    ('Logistic Regression', lr_pipeline),
    ('k-NN', knn_pipeline),
    ('Linear SVM', best_svm_linear),
    ('RBF SVM', best_svm_rbf)
]):
    ax = axes[idx // 2, idx % 2]
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'{name} - Confusion Matrix', fontsize=11)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(['0', '1'])
    ax.set_yticklabels(['0', '1'])

plt.tight_layout()
fig.savefig(os.path.join(figures_dir, 'confusion_matrices_all.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrices_all.png")

# ============================================================
# 6. ROC CURVE COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("GENERATING ROC CURVE COMPARISON")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
colors = ['#2b5c8f', '#d9534f', '#5cb85c', '#f0ad4e']
for idx, (name, pipeline) in enumerate([
    ('Logistic Regression', lr_pipeline),
    ('k-NN', knn_pipeline),
    ('Linear SVM', best_svm_linear),
    ('RBF SVM', best_svm_rbf)
]):
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=colors[idx], lw=2, label=f'{name} (AUC = {auc:.4f})')

ax.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Random (AUC = 0.50)')
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
ax.set_title('ROC Curve Comparison - All Models (Test Set)', fontsize=12, pad=15)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(figures_dir, 'roc_curve_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: roc_curve_comparison.png")

# ============================================================
# 7. LEARNING CURVES
# ============================================================

print("\n" + "=" * 60)
print("GENERATING LEARNING CURVES")
print("=" * 60)

train_sizes = np.linspace(0.1, 1.0, 6)

def plot_learning_curve(estimator, title, X, y, cv, train_sizes, save_path):
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='roc_auc',
        train_sizes=train_sizes, n_jobs=-1, shuffle=True, random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(train_sizes_abs, train_mean, 'o-', color='#2b5c8f', label='Training Score')
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2, color='#2b5c8f')
    ax.plot(train_sizes_abs, test_mean, 'o-', color='#d9534f', label='Cross-Validation Score')
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.2, color='#d9534f')
    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('ROC-AUC Score', fontsize=11)
    ax.set_title(title, fontsize=12, pad=15)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return train_sizes_abs, train_scores, test_scores

for name, pipeline, save_name in [
    ('Logistic Regression', lr_pipeline, 'baseline_learning_curve.png'),
    ('k-NN', knn_pipeline, 'knn_learning_curve.png'),
    ('Linear SVM', best_svm_linear, 'svm_linear_learning_curve.png'),
    ('RBF SVM', best_svm_rbf, 'svm_rbf_learning_curve.png')
]:
    print(f"  {name}...")
    plot_learning_curve(
        pipeline,
        f'Learning Curve - {name}',
        X_train, y_train, cv, train_sizes,
        os.path.join(figures_dir, save_name)
    )
print("Saved all learning curves!")

# ============================================================
# 8. VALIDATION CURVES
# ============================================================

print("\n" + "=" * 60)
print("GENERATING VALIDATION CURVES")
print("=" * 60)

def plot_validation_curve(estimator, title, X, y, param_name, param_range, cv, save_path):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring='roc_auc', n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(param_range, train_mean, 'o-', color='#2b5c8f', label='Training Score')
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color='#2b5c8f')
    ax.plot(param_range, test_mean, 'o-', color='#d9534f', label='Cross-Validation Score')
    ax.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color='#d9534f')
    
    xlabel = param_name.replace('classifier__', '').replace('estimator__', '')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('ROC-AUC Score', fontsize=11)
    ax.set_title(title, fontsize=12, pad=15)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    if all(isinstance(x, (int, float)) and x > 0 for x in param_range):
        if max(param_range) / min(param_range) > 10:
            ax.set_xscale('log')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return train_scores, test_scores

# 8.1 k-NN: n_neighbors
print("  k-NN: n_neighbors...")
knn_param_range = [1, 3, 5, 7, 9, 11, 13, 15, 19, 25, 31]
knn_pipeline_for_val = Pipeline(steps=[
    ('preprocessor', get_preprocessor_pipeline()),
    ('classifier', KNeighborsClassifier(weights='distance', p=2, metric='minkowski'))
])
plot_validation_curve(
    knn_pipeline_for_val,
    'Validation Curve - k-NN: n_neighbors',
    X_train, y_train, 'classifier__n_neighbors', knn_param_range, cv,
    os.path.join(figures_dir, 'knn_validation_curve.png')
)

# 8.2 Linear SVM: C
print("  Linear SVM: C...")
svm_linear_param_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
svm_linear_pipeline_for_val = Pipeline(steps=[
    ('preprocessor', get_preprocessor_pipeline()),
    ('classifier', CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=5000), cv=3, method='sigmoid'))
])
plot_validation_curve(
    svm_linear_pipeline_for_val,
    'Validation Curve - Linear SVM: C',
    X_train, y_train, 'classifier__estimator__C', svm_linear_param_range, cv,
    os.path.join(figures_dir, 'svm_linear_validation_curve.png')
)

# 8.3 RBF SVM: C (with best gamma)
print("  RBF SVM: C...")
best_gamma = joblib.load(os.path.join(results_dir, 'svm_rbf_best_params.csv'))['classifier__gamma'].iloc[0]
svm_rbf_pipeline_for_val = Pipeline(steps=[
    ('preprocessor', get_preprocessor_pipeline()),
    ('classifier', SVC(kernel='rbf', gamma=best_gamma, probability=True, random_state=42))
])
svm_rbf_param_range = [0.01, 0.1, 1, 10, 100, 1000]
plot_validation_curve(
    svm_rbf_pipeline_for_val,
    f'Validation Curve - RBF SVM: C (gamma={best_gamma})',
    X_train, y_train, 'classifier__C', svm_rbf_param_range, cv,
    os.path.join(figures_dir, 'svm_rbf_validation_curve.png')
)

print("Saved all validation curves!")

# ============================================================
# 9. SAVE SUMMARY JSON
# ============================================================

print("\n" + "=" * 60)
print("SAVING SUMMARY")
print("=" * 60)

summary = {
    'cv_results': {},
    'test_results': test_metrics,
    'best_params': {
        'Logistic Regression': {'class_weight': 'balanced', 'random_state': 42, 'max_iter': 1000},
        'k-NN': best_knn_params,
        'Linear SVM': pd.read_csv(os.path.join(results_dir, 'svm_linear_best_params.csv')).iloc[0].to_dict(),
        'RBF SVM': pd.read_csv(os.path.join(results_dir, 'svm_rbf_best_params.csv')).iloc[0].to_dict()
    }
}

for name, res in models_cv.items():
    summary['cv_results'][name] = {}
    for metric in ['roc_auc', 'recall', 'f1', 'precision', 'accuracy']:
        scores = res[f'test_{metric}']
        summary['cv_results'][name][metric] = {'mean': float(scores.mean()), 'std': float(scores.std())}

with open(os.path.join(results_dir, 'experiment_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("Saved: experiment_summary.json")
print("\n" + "=" * 60)
print("ALL EXPERIMENTS COMPLETE!")
print("=" * 60)