import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.features import load_and_split_data, get_preprocessor_pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score, accuracy_score

# Load data
data_path = str(Path(__file__).resolve().parent / 'data' / 'framingham.csv')
X_train, X_test, y_train, y_test = load_and_split_data(data_path=data_path, test_size=0.2, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = 'roc_auc'

print("=== LINEAR SVM GRIDSEARCH ===")
svm_linear_pipeline = Pipeline(steps=[
    ('preprocessor', get_preprocessor_pipeline()),
    ('classifier', SVC(kernel='linear', probability=True, random_state=42))
])

param_grid_svm_linear = {'classifier__C': [0.01, 0.1, 1, 10, 100], 'classifier__class_weight': [None, 'balanced']}

start = time.time()
svm_linear_grid = GridSearchCV(
    estimator=svm_linear_pipeline,
    param_grid=param_grid_svm_linear,
    scoring=scoring, cv=cv, n_jobs=-1, refit=True,
    return_train_score=True, verbose=1
)
svm_linear_grid.fit(X_train, y_train)
svm_linear_time = time.time() - start

print(f"Time: {svm_linear_time:.1f}s")
print(f"Best params: {svm_linear_grid.best_params_}")
print(f"Best CV ROC-AUC: {svm_linear_grid.best_score_:.4f}")

# Save results
svm_linear_results_df = pd.DataFrame(svm_linear_grid.cv_results_)
svm_linear_results_df.to_csv(str(Path(__file__).resolve().parent / 'models' / 'results' / 'svm_linear_gridsearch_results.csv'), index=False)
print("Saved: svm_linear_gridsearch_results.csv")

# Test set evaluation
best_svm_linear = svm_linear_grid.best_estimator_
svm_linear_y_pred = best_svm_linear.predict(X_test)
svm_linear_y_prob = best_svm_linear.predict_proba(X_test)[:, 1]

svm_linear_test_metrics = {
    'ROC-AUC': roc_auc_score(y_test, svm_linear_y_prob),
    'Recall': recall_score(y_test, svm_linear_y_pred),
    'F1-Score': f1_score(y_test, svm_linear_y_pred),
    'Precision': precision_score(y_test, svm_linear_y_pred),
    'Accuracy': accuracy_score(y_test, svm_linear_y_pred)
}
print("Test metrics:", svm_linear_test_metrics)

# Save best model and metrics for later use
import joblib
joblib.dump(best_svm_linear, str(Path(__file__).resolve().parent / 'models' / 'best_svm_linear.joblib'))
pd.DataFrame([svm_linear_test_metrics]).to_csv(str(Path(__file__).resolve().parent / 'models' / 'results' / 'svm_linear_test_metrics.csv'), index=False)
pd.DataFrame([svm_linear_grid.best_params_]).to_csv(str(Path(__file__).resolve().parent / 'models' / 'results' / 'svm_linear_best_params.csv'), index=False)

print("=== RBF SVM GRIDSEARCH ===")
svm_rbf_pipeline = Pipeline(steps=[
    ('preprocessor', get_preprocessor_pipeline()),
    ('classifier', SVC(kernel='rbf', probability=True, random_state=42))
])

# Reduced parameter grid for faster execution
param_grid_svm_rbf = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 0.01, 0.1, 1],
    'classifier__class_weight': [None, 'balanced']
}

total = len(param_grid_svm_rbf['classifier__C']) * len(param_grid_svm_rbf['classifier__gamma'])
print(f"Total combinations: {total}")

start = time.time()
svm_rbf_grid = GridSearchCV(
    estimator=svm_rbf_pipeline,
    param_grid=param_grid_svm_rbf,
    scoring=scoring, cv=cv, n_jobs=-1, refit=True,
    return_train_score=True, verbose=1
)
svm_rbf_grid.fit(X_train, y_train)
svm_rbf_time = time.time() - start

print(f"Time: {svm_rbf_time:.1f}s")
print(f"Best params: {svm_rbf_grid.best_params_}")
print(f"Best CV ROC-AUC: {svm_rbf_grid.best_score_:.4f}")

# Save results
svm_rbf_results_df = pd.DataFrame(svm_rbf_grid.cv_results_)
svm_rbf_results_df.to_csv(str(Path(__file__).resolve().parent / 'models' / 'results' / 'svm_rbf_gridsearch_results.csv'), index=False)
print("Saved: svm_rbf_gridsearch_results.csv")

# Test set evaluation
best_svm_rbf = svm_rbf_grid.best_estimator_
svm_rbf_y_pred = best_svm_rbf.predict(X_test)
svm_rbf_y_prob = best_svm_rbf.predict_proba(X_test)[:, 1]

svm_rbf_test_metrics = {
    'ROC-AUC': roc_auc_score(y_test, svm_rbf_y_prob),
    'Recall': recall_score(y_test, svm_rbf_y_pred),
    'F1-Score': f1_score(y_test, svm_rbf_y_pred),
    'Precision': precision_score(y_test, svm_rbf_y_pred),
    'Accuracy': accuracy_score(y_test, svm_rbf_y_pred)
}
print("Test metrics:", svm_rbf_test_metrics)

# Save
joblib.dump(best_svm_rbf, str(Path(__file__).resolve().parent / 'models' / 'best_svm_rbf.joblib'))
pd.DataFrame([svm_rbf_test_metrics]).to_csv(str(Path(__file__).resolve().parent / 'models' / 'results' / 'svm_rbf_test_metrics.csv'), index=False)
pd.DataFrame([svm_rbf_grid.best_params_]).to_csv(str(Path(__file__).resolve().parent / 'models' / 'results' / 'svm_rbf_best_params.csv'), index=False)

print("\n=== ALL SVM EXPERIMENTS COMPLETE ===")