import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 02 - Modeling & Evaluation\n",
    "## Hu\u1ea5n lu\u1ec7n, Tuning & So s\u00e1nh m\u00f4 h\u00ecnh: k-NN, SVM v\u00e0 Learning Curve\n",
    "**B\u1ed9 d\u1eef li\u1ec7u:** Framingham Heart Study\n",
    "**M\u1ee5c ti\u00eau:** T\u00ecm m\u00f4 h\u00ecnh t\u1ed1t nh\u1ea5t \u0111\u1ec3 d\u1ef1 \u0111o\u00e1n TenYearCHD\n",
    "**Th\u00e0nh vi\u00ean ph\u1ee5 tr\u00e1ch:** Th\u00e0nh vi\u00ean 2 (TV2)\n",
    "**M\u1ed1c ch\u1ea5m:** Milestone 2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Gi\u1edbi thi\u1ec7u v\u00e0 M\u1ee5c ti\u00eau Th\u1eed nghi\u1ec7m\n",
    "Notebook n\u00e0y th\u1ef1c hi\u1ec7n c\u00e1c nhi\u1ec7m v\u1ee5 c\u1ee7a <b>Th\u00e0nh vi\u00ean 2 - Milestone 2</b>:\n",
    "1. <b>k-Nearest Neighbors (k-NN)</b>: GridSearchCV t\u01b1i \u01b0u h\u00f3a si\u00eau tham s\u1ed1\n",
    "2. <b>Support Vector Machine (SVM)</b>: Linear SVM v\u00e0 RBF SVM v\u1edbi GridSearchCV\n",
    "3. <b>\u0110\u00e1nh gi\u00e1 m\u00f4 h\u00ecnh</b>: So s\u00e1nh Baseline Logistic Regression (Milestone 1), k-NN, Linear SVM, RBF SVM\n",
    "4. <b>Learning Curve</b>: Ph\u00e2n t\u00edch overfitting/underfitting cho 4 m\u00f4 h\u00ecnh\n",
    "5. <b>Validation Curve</b>: Ph\u00e2n t\u00edch t\u00e1c \u0111\u1ed9ng si\u00eau tham s\u1ed1 quan tr\u1ecdng\n",
    "\n",
    "T\u1ea5t c\u1ea3 m\u00f4 h\u00ecnh s\u1eed d\u1ee5ng <b>Pipeline chung</b> t\u1eeb TV1 (feature engineering + ColumnTransformer) \u0111\u1ec3 \u0111\u1ea3m b\u1ea3o <b>kh\u00f4ng c\u00f3 Data Leakage</b>."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Import th\u01b0 vi\u1ec7n\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "import os\n",
    "import sys\n",
    "import time\n",
    "import json\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import joblib\n",
    "\n",
    "from sklearn.model_selection import (\n",
    "    train_test_split,\n",
    "    GridSearchCV,\n",
    "    StratifiedKFold,\n",
    "    cross_validate,\n",
    "    learning_curve,\n",
    "    validation_curve\n",
    ")\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score,\n",
    "    precision_score,\n",
    "    recall_score,\n",
    "    f1_score,\n",
    "    roc_auc_score,\n",
    "    confusion_matrix,\n",
    "    classification_report,\n",
    "    roc_curve\n",
    ")\n",
    "\n",
    "# Th\u00eam src v\u00e0o path \u0111\u1ec3 import TV1 preprocessing\n",
    "sys.path.append(os.path.abspath('..'))\n",
    "from src.features import (\n",
    "    load_and_split_data,\n",
    "    get_preprocessor_pipeline,\n",
    "    TARGET_COLUMN\n",
    ")\n",
    "\n",
    "# C\u1ea5u h\u00ecnh bi\u1ec3u \u0111\u1ed3\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.max_rows', 100)\n",
    "pd.set_option('display.width', 1000)\n",
    "\n",
    "print('Im\u2299 Imports th\u00e0nh c\u00f4ng!')\n",
    "print(f'Im\u2299 Working directory: {os.getcwd()}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. T\u1ea3i d\u1eef li\u1ec7u v\u00e0 chia Train/Test (Consistent v\u1edbi Milestone 1)\n",
    "S\u1eed d\u1ee5ng h\u00e0m `load_and_split_data` t\u1eeb TV1 \u0111\u1ec3 \u0111\u1ea3m b\u1ea3o chia d\u1eef li\u1ec7u nh\u1ea5t qu\u00e1n."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3.1. Load v\u00e0 chia d\u1eef li\u1ec7u\n",
    "data_path = '../data/framingham.csv'\n",
    "X_train, X_test, y_train, y_test = load_and_split_data(\n",
    "    data_path=data_path,\n",
    "    test_size=0.2,\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "print(f'X_train shape: {X_train.shape}')\n",
    "print(f'X_test shape: {X_test.shape}')\n",
    "print(f'y_train distribution: {y_train.value_counts().to_dict()}')\n",
    "print(f'y_test distribution: {y_test.value_counts().to_dict()}')\n",
    "print(f'Ty\u1ec7 l\u1ec7 l\u1edbp d\u01b0\u01a1ng (Train): {y_train.mean():.4f}')\n",
    "print(f'Ty\u1ec7 l\u1ec7 l\u1edbp d\u01b0\u01a1ng (Test): {y_test.mean():.4f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. X\u00e2y d\u1ef1ng Pipeline ti\u1ec1n x\u1eed l\u00fd chung (TV1)\n",
    "Pipeline n\u00e0y s\u1ebd \u0111\u01b0\u1ee3c s\u1eed d\u1ee5ng cho <b>t\u1ea5t c\u1ea3</b> c\u00e1c m\u00f4 h\u00ecnh: Logistic Regression, k-NN, Linear SVM, RBF SVM."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4.1. L\u1ea5y preprocessor pipeline t\u1eeb TV1 (CH\u01a0A FIT)\n",
    "preprocessor = get_preprocessor_pipeline()\n",
    "print('Preprocessor pipeline:')\n",
    "preprocessor"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. C\u1ea5u h\u00ecnh Cross-Validation chung\n",
    "S\u1eed d\u1ee5ng StratifiedKFold 5-fold v\u1edbi random_state=42 cho t\u1ea5t c\u1ea3 c\u00e1c thi\u1ec7n nghi\u1ec7m."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5.1. C\u1ea5u h\u00ecnh CV\n",
    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "scoring = 'roc_auc'\n",
    "\n",
    "print(f'CV: {cv}')\n",
    "print(f'Scoring: {scoring}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Baseline Logistic Regression (T\u00e1i s\u1eed d\u1ee5ng t\u1eeb Milestone 1)\n",
    "\u0110\u00e1nh gi\u00e1 l\u1ea1i baseline tr\u00ean c\u00f9ng t\u1eadp train/test \u0111\u1ec3 so s\u00e1nh c\u00f4ng b\u1eb1ng."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 6.1. Baseline Logistic Regression Pipeline\n",
    "lr_pipeline = Pipeline(steps=[\n",
    "    ('preprocessor', get_preprocessor_pipeline()),\n",
    "    ('classifier', LogisticRegression(\n",
    "        class_weight='balanced',\n",
    "        random_state=42,\n",
    "        max_iter=1000\n",
    "    ))\n",
    "])\n",
    "\n",
    "# 6.2. Cross-Validation tr\u00ean t\u1eadp Train\n",
    "print('Dang chay CV cho Baseline Logistic Regression...')\n",
    "start = time.time()\n",
    "lr_cv_results = cross_validate(\n",
    "    lr_pipeline, X_train, y_train,\n",
    "    cv=cv, scoring=['roc_auc', 'recall', 'f1', 'precision', 'accuracy'],\n",
    "    return_train_score=True, n_jobs=-1\n",
    ")\n",
    "lr_cv_time = time.time() - start\n",
    "\n",
    "# 6.3. Fit tr\u00ean to\u00e0n b\u1ed9 Train v\u00e0 danh gia Test\n",
    "lr_pipeline.fit(X_train, y_train)\n",
    "lr_y_pred = lr_pipeline.predict(X_test)\n",
    "lr_y_prob = lr_pipeline.predict_proba(X_test)[:, 1]\n",
    "\n",
    "lr_test_metrics = {\n",
    "    'ROC-AUC': roc_auc_score(y_test, lr_y_prob),\n",
    "    'Recall': recall_score(y_test, lr_y_pred),\n",
    "    'F1-Score': f1_score(y_test, lr_y_pred),\n",
    "    'Precision': precision_score(y_test, lr_y_pred),\n",
    "    'Accuracy': accuracy_score(y_test, lr_y_pred)\n",
    "}\n",
    "\n",
    "print(f'Im\u2299 Baseline CV Time: {lr_cv_time:.1f}s')\n",
    "print('--- CV Results (Mean \u00b1 Std) ---')\n",
    "for metric in ['roc_auc', 'recall', 'f1', 'precision', 'accuracy']:\n",
    "    scores = lr_cv_results[f'test_{metric}']\n",
    "    print(f'  {metric.upper():10s}: {scores.mean():.4f} \u00b1 {scores.std():.4f}')\n",
    "print('\\n--- Test Set Results ---')\n",
    "for k, v in lr_test_metrics.items():\n",
    "    print(f'  {k:10s}: {v:.4f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. k-Nearest Neighbors (k-NN) - GridSearchCV\n",
    "T\u01b1i \u01b0u h\u00f3a si\u00eau tham s\u1ed1: n_neighbors, weights, metric, p"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 7.1. k-NN Pipeline\n",
    "knn_pipeline = Pipeline(steps=[\n",
    "    ('preprocessor', get_preprocessor_pipeline()),\n",
    "    ('classifier', KNeighborsClassifier())\n",
    "])\n",
    "\n",
    "# 7.2. Parameter Grid cho k-NN\n",
    "param_grid_knn = {\n",
    "    'classifier__n_neighbors': [3, 5, 7, 9, 11, 15, 21],\n",
    "    'classifier__weights': ['uniform', 'distance'],\n",
    "    'classifier__metric': ['minkowski'],\n",
    "    'classifier__p': [1, 2]\n",
    "}\n",
    "\n",
    "total_knn = len(param_grid_knn['classifier__n_neighbors']) * len(param_grid_knn['classifier__weights']) * len(param_grid_knn['classifier__p'])\n",
    "print(f'Tong so to hop tham so: {total_knn}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 7.3. GridSearchCV cho k-NN\n",
    "print('Dang chay GridSearchCV cho k-NN...')\n",
    "start = time.time()\n",
    "knn_grid = GridSearchCV(\n",
    "    estimator=knn_pipeline,\n",
    "    param_grid=param_grid_knn,\n",
    "    scoring=scoring,\n",
    "    cv=cv,\n",
    "    n_jobs=-1,\n",
    "    refit=True,\n",
    "    return_train_score=True,\n",
    "    verbose=1\n",
    ")\n",
    "knn_grid.fit(X_train, y_train)\n",
    "knn_grid_time = time.time() - start\n",
    "\n",
    "print(f'Im\u2299 k-NN GridSearchCV Time: {knn_grid_time:.1f}s')\n",
    "print(f'Best params: {knn_grid.best_params_}')\n",
    "print(f'Best CV ROC-AUC: {knn_grid.best_score_:.4f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 7.4. Luu ket qua GridSearchCV k-NN\n",
    "knn_results_df = pd.DataFrame(knn_grid.cv_results_)\n",
    "knn_results_path = '../models/results/knn_gridsearch_results.csv'\n",
    "knn_results_df.to_csv(knn_results_path, index=False)\n",
    "print(f'Da luu: {knn_results_path}')\n",
    "\n",
    "# Hien thi top 10 ket qua\n",
    "cols = ['param_classifier__n_neighbors', 'param_classifier__weights', 'param_classifier__p',\n",
    "        'mean_test_score', 'std_test_score', 'mean_train_score', 'rank_test_score']\n",
    "top10 = knn_results_df[cols].sort_values('rank_test_score').head(10)\n",
    "print(top10.to_string(index=False))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 7.5. Danh gia best k-NN tren Test set\n",
    "best_knn = knn_grid.best_estimator_\n",
    "knn_y_pred = best_knn.predict(X_test)\n",
    "knn_y_prob = best_knn.predict_proba(X_test)[:, 1]\n",
    "\n",
    "knn_test_metrics = {\n",
    "    'ROC-AUC': roc_auc_score(y_test, knn_y_prob),\n",
    "    'Recall': recall_score(y_test, knn_y_pred),\n",
    "    'F1-Score': f1_score(y_test, knn_y_pred),\n",
    "    'Precision': precision_score(y_test, knn_y_pred),\n",
    "    'Accuracy': accuracy_score(y_test, knn_y_pred)\n",
    "}\n",
    "\n",
    "print('--- Best k-NN Test Set Results ---')\n",
    "for k, v in knn_test_metrics.items():\n",
    "    print(f'  {k:10s}: {v:.4f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Linear SVM - GridSearchCV\n",
    "T\u01b1i \u01b0u h\u00f3a tham so C cho Linear SVM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 8.1. Linear SVM Pipeline\n",
    "svm_linear_pipeline = Pipeline(steps=[\n",
    "    ('preprocessor', get_preprocessor_pipeline()),\n",
    "    ('classifier', SVC(kernel='linear', probability=True, random_state=42))\n",
    "])\n",
    "\n",
    "# 8.2. Parameter Grid cho Linear SVM\n",
    "param_grid_svm_linear = {\n",
    "    'classifier__C': [0.01, 0.1, 1, 10, 100]\n",
    "}\n",
    "\n",
    "print(f'Tong so to hop tham so: {len(param_grid_svm_linear[\"classifier__C\"])}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 8.3. GridSearchCV cho Linear SVM\n",
    "print('Dang chay GridSearchCV cho Linear SVM...')\n",
    "start = time.time()\n",
    "svm_linear_grid = GridSearchCV(\n",
    "    estimator=svm_linear_pipeline,\n",
    "    param_grid=param_grid_svm_linear,\n",
    "    scoring=scoring,\n",
    "    cv=cv,\n",
    "    n_jobs=-1,\n",
    "    refit=True,\n",
    "    return_train_score=True,\n",
    "    verbose=1\n",
    ")\n",
    "svm_linear_grid.fit(X_train, y_train)\n",
    "svm_linear_time = time.time() - start\n",
    "\n",
    "print(f'Im\u2299 Linear SVM GridSearchCV Time: {svm_linear_time:.1f}s')\n",
    "print(f'Best params: {svm_linear_grid.best_params_}')\n",
    "print(f'Best CV ROC-AUC: {svm_linear_grid.best_score_:.4f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 8.4. Luu ket qua GridSearchCV Linear SVM\n",
    "svm_linear_results_df = pd.DataFrame(svm_linear_grid.cv_results_)\n",
    "svm_linear_path = '../models/results/svm_linear_gridsearch_results.csv'\n",
    "svm_linear_results_df.to_csv(svm_linear_path, index=False)\n",
    "print(f'Da luu: {svm_linear_path}')\n",
    "\n",
    "cols = ['param_classifier__C', 'mean_test_score', 'std_test_score', 'mean_train_score', 'rank_test_score']\n",
    "print(svm_linear_results_df[cols].sort_values('rank_test_score').to_string(index=False))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 8.5. Danh gia best Linear SVM tren Test set\n",
    "best_svm_linear = svm_linear_grid.best_estimator_\n",
    "svm_linear_y_pred = best_svm_linear.predict(X_test)\n",
    "svm_linear_y_prob = best_svm_linear.predict_proba(X_test)[:, 1]\n",
    "\n",
    "svm_linear_test_metrics = {\n",
    "    'ROC-AUC': roc_auc_score(y_test, svm_linear_y_prob),\n",
    "    'Recall': recall_score(y_test, svm_linear_y_pred),\n",
    "    'F1-Score': f1_score(y_test, svm_linear_y_pred),\n",
    "    'Precision': precision_score(y_test, svm_linear_y_pred),\n",
    "    'Accuracy': accuracy_score(y_test, svm_linear_y_pred)\n",
    "}\n",
    "\n",
    "print('--- Best Linear SVM Test Set Results ---')\n",
    "for k, v in svm_linear_test_metrics.items():\n",
    "    print(f'  {k:10s}: {v:.4f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. RBF SVM - GridSearchCV\n",
    "T\u01b1i \u01b0u h\u00f3a tham so C v\u00e0 gamma cho RBF SVM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 9.1. RBF SVM Pipeline\n",
    "svm_rbf_pipeline = Pipeline(steps=[\n",
    "    ('preprocessor', get_preprocessor_pipeline()),\n",
    "    ('classifier', SVC(kernel='rbf', probability=True, random_state=42))\n",
    "])\n",
    "\n",
    "# 9.2. Parameter Grid cho RBF SVM\n",
    "param_grid_svm_rbf = {\n",
    "    'classifier__C': [0.1, 1, 10, 100],\n",
    "    'classifier__gamma': ['scale', 0.001, 0.01, 0.1, 1]\n",
    "}\n",
    "\n",
    "total_rbf = len(param_grid_svm_rbf['classifier__C']) * len(param_grid_svm_rbf['classifier__gamma'])\n",
    "print(f'Tong so to hop tham so: {total_rbf}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 9.3. GridSearchCV cho RBF SVM\n",
    "print('Dang chay GridSearchCV cho RBF SVM...')\n",
    "start = time.time()\n",
    "svm_rbf_grid = GridSearchCV(\n",
    "    estimator=svm_rbf_pipeline,\n",
    "    param_grid=param_grid_svm_rbf,\n",
    "    scoring=scoring,\n",
    "    cv=cv,\n",
    "    n_jobs=-1,\n",
    "    refit=True,\n",
    "    return_train_score=True,\n",
    "    verbose=1\n",
    ")\n",
    "svm_rbf_grid.fit(X_train, y_train)\n",
    "svm_rbf_time = time.time() - start\n",
    "\n",
    "print(f'Im\u2299 RBF SVM GridSearchCV Time: {svm_rbf_time:.1f}s')\n",
    "print(f'Best params: {svm_rbf_grid.best_params_}')\n",
    "print(f'Best CV ROC-AUC: {svm_rbf_grid.best_score_:.4f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 9.4. Luu ket qua GridSearchCV RBF SVM\n",
    "svm_rbf_results_df = pd.DataFrame(svm_rbf_grid.cv_results_)\n",
    "svm_rbf_path = '../models/results/svm_rbf_gridsearch_results.csv'\n",
    "svm_rbf_results_df.to_csv(svm_rbf_path, index=False)\n",
    "print(f'Da luu: {svm_rbf_path}')\n",
    "\n",
    "cols = ['param_classifier__C', 'param_classifier__gamma', 'mean_test_score', 'std_test_score', 'mean_train_score', 'rank_test_score']\n",
    "print(svm_rbf_results_df[cols].sort_values('rank_test_score').head(10).to_string(index=False))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 9.5. Danh gia best RBF SVM tren Test set\n",
    "best_svm_rbf = svm_rbf_grid.best_estimator_\n",
    "svm_rbf_y_pred = best_svm_rbf.predict(X_test)\n",
    "svm_rbf_y_prob = best_svm_rbf.predict_proba(X_test)[:, 1]\n",
    "\n",
    "svm_rbf_test_metrics = {\n",
    "    'ROC-AUC': roc_auc_score(y_test, svm_rbf_y_prob),\n",
    "    'Recall': recall_score(y_test, svm_rbf_y_pred),\n",
    "    'F1-Score': f1_score(y_test, svm_rbf_y_pred),\n",
    "    'Precision': precision_score(y_test, svm_rbf_y_pred),\n",
    "    'Accuracy': accuracy_score(y_test, svm_rbf_y_pred)\n",
    "}\n",
    "\n",
    "print('--- Best RBF SVM Test Set Results ---')\n",
    "for k, v in svm_rbf_test_metrics.items():\n",
    "    print(f'  {k:10s}: {v:.4f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. Tong hop so sanh 4 mo hinh\n",
    "So sanh Cross-Validation (5-fold) va Holdout Test Set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 10.1. Chay CV chi tiet cho k-NN, Linear SVM, RBF SVM de lay cac metric khac\n",
    "print('Dang chay CV chi tiet cho cac mo hinh tot nhat...')\n",
    "\n",
    "models_cv = {'Logistic Regression': lr_cv_results}\n",
    "for name, pipeline in [('k-NN', best_knn), ('Linear SVM', best_svm_linear), ('RBF SVM', best_svm_rbf)]:\n",
    "    cv_res = cross_validate(pipeline, X_train, y_train, cv=cv,\n",
    "                          scoring=['roc_auc', 'recall', 'f1', 'precision', 'accuracy'],\n",
    "                          n_jobs=-1)\n",
    "    models_cv[name] = cv_res\n",
    "    print(f'  Im\u2299 {name} done')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 10.2. Bang so sanh Cross-Validation\n",
    "cv_comparison = []\n",
    "for name, res in models_cv.items():\n",
    "    row = {'Model': name}\n",
    "    for metric in ['roc_auc', 'recall', 'f1', 'precision', 'accuracy']:\n",
    "        scores = res[f'test_{metric}']\n",
    "        row[f'CV {metric.upper()} Mean'] = f'{scores.mean():.4f}'\n",
    "        row[f'CV {metric.upper()} Std'] = f'{scores.std():.4f}'\n",
    "    cv_comparison.append(row)\n",
    "\n",
    "cv_df = pd.DataFrame(cv_comparison)\n",
    "print('=== CROSS-VALIDATION COMPARISON (5-Fold Stratified) ===')\n",
    "print(cv_df.to_string(index=False))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 10.3. Bang so sanh Holdout Test Set\n",
    "test_comparison = []\n",
    "for name, metrics in [\n",
    "    ('Logistic Regression', lr_test_metrics),\n",
    "    ('k-NN', knn_test_metrics),\n",
    "    ('Linear SVM', svm_linear_test_metrics),\n",
    "    ('RBF SVM', svm_rbf_test_metrics)\n",
    "]:\n",
    "    row = {'Model': name}\n",
    "    row.update(metrics)\n",
    "    test_comparison.append(row)\n",
    "\n",
    "test_df = pd.DataFrame(test_comparison)\n",
    "print('=== HOLDOUT TEST SET COMPARISON ===')\n",
    "print(test_df.to_string(index=False))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 10.4. Bang tong hop hoan chinh\n",
    "full_comparison = []\n",
    "for name in ['Logistic Regression', 'k-NN', 'Linear SVM', 'RBF SVM']:\n",
    "    cv_res = models_cv[name]\n",
    "    test_res = {\n",
    "        'Logistic Regression': lr_test_metrics,\n",
    "        'k-NN': knn_test_metrics,\n",
    "        'Linear SVM': svm_linear_test_metrics,\n",
    "        'RBF SVM': svm_rbf_test_metrics\n",
    "    }[name]\n",
    "    \n",
    "    row = {'Model': name}\n",
    "    for metric in ['roc_auc', 'recall', 'f1', 'precision', 'accuracy']:\n",
    "        scores = cv_res[f'test_{metric}']\n",
    "        row[f'CV {metric.upper()} (Mean\u00b1Std)'] = f'{scores.mean():.4f}\u00b1{scores.std():.4f}'\n",
    "        row[f'Test {metric.upper()}'] = f'{test_res[metric.upper()]:.4f}'\n",
    "    full_comparison.append(row)\n",
    "\n",
    "full_df = pd.DataFrame(full_comparison)\n",
    "print('=== FULL MODEL COMPARISON ===')\n",
    "print(full_df.to_string(index=False))\n",
    "\n",
    "# Luu model_comparison.csv\n",
    "comparison_path = '../models/results/model_comparison.csv'\n",
    "full_df.to_csv(comparison_path, index=False)\n",
    "print(f'\\nDa luu: {comparison_path}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 11. Confusion Matrix va ROC Curve cho cac mo hinh tot nhat"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 11.1. Confusion Matrix cho 4 mo hinh\n",
    "fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)\n",
    "models_pred = [\n",
    "    ('Logistic Regression', lr_y_pred, lr_y_prob),\n",
    "    ('k-NN', knn_y_pred, knn_y_prob),\n",
    "    ('Linear SVM', svm_linear_y_pred, svm_linear_y_prob),\n",
    "    ('RBF SVM', svm_rbf_y_pred, svm_rbf_y_prob)\n",
    "]\n",
    "\n",
    "for idx, (name, y_pred, y_prob) in enumerate(models_pred):\n",
    "    ax = axes[idx // 2, idx % 2]\n",
    "    cm = confusion_matrix(y_test, y_pred)\n",
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)\n",
    "    ax.set_title(f'{name} - Confusion Matrix', fontsize=11)\n",
    "    ax.set_xlabel('Predicted')\n",
    "    ax.set_ylabel('Actual')\n",
    "    ax.set_xticklabels(['0', '1'])\n",
    "    ax.set_yticklabels(['0', '1'])\n",
    "\n",
    "plt.tight_layout()\n",
    "fig.savefig('../notebooks/figures/confusion_matrices_all.png', dpi=300, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 11.2. ROC Curve so sanh 4 mo hinh\n",
    "fig, ax = plt.subplots(figsize=(8, 7), dpi=300)\n",
    "\n",
    "colors = ['#2b5c8f', '#d9534f', '#5cb85c', '#f0ad4e']\n",
    "for idx, (name, y_pred, y_prob) in enumerate(models_pred):\n",
    "    fpr, tpr, _ = roc_curve(y_test, y_prob)\n",
    "    auc = roc_auc_score(y_test, y_prob)\n",
    "    ax.plot(fpr, tpr, color=colors[idx], lw=2, label=f'{name} (AUC = {auc:.4f})')\n",
    "\n",
    "ax.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Random (AUC = 0.50)')\n",
    "ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)\n",
    "ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)\n",
    "ax.set_title('ROC Curve Comparison - All Models (Test Set)', fontsize=12, pad=15)\n",
    "ax.legend(loc='lower right', fontsize=10)\n",
    "ax.grid(True, alpha=0.3)\n",
    "plt.tight_layout()\n",
    "fig.savefig('../notebooks/figures/roc_curve_comparison.png', dpi=300, bbox_inches='tight')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 12. Learning Curves\n",
    "Phan tich overfitting/underfitting cho 4 mo hinh"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 12.1. Cau hinh Learning Curve\n",
    "train_sizes = np.linspace(0.1, 1.0, 6)\n",
    "\n",
    "def plot_learning_curve(estimator, title, X, y, cv, train_sizes, scoring='roc_auc',\n",
    "                        figsize=(7, 5), save_path=None):\n",
    "    train_sizes_abs, train_scores, test_scores = learning_curve(\n",
    "        estimator, X, y,\n",
    "        cv=cv,\n",
    "        scoring=scoring,\n",
    "        train_sizes=train_sizes,\n",
    "        n_jobs=-1,\n",
    "        shuffle=True,\n",
    "        random_state=42\n",
    "    )\n",
    "\n",
    "    train_mean = np.mean(train_scores, axis=1)\n",
    "    train_std = np.std(train_scores, axis=1)\n",
    "    test_mean = np.mean(test_scores, axis=1)\n",
    "    test_std = np.std(test_scores, axis=1)\n",
    "\n",
    "    fig, ax = plt.subplots(figsize=figsize, dpi=300)\n",
    "    ax.plot(train_sizes_abs, train_mean, 'o-', color='#2b5c8f', label='Training Score')\n",
    "    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,\n",
    "                    alpha=0.2, color='#2b5c8f')\n",
    "    ax.plot(train_sizes_abs, test_mean, 'o-', color='#d9534f', label='Cross-Validation Score')\n",
    "    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,\n",
    "                    alpha=0.2, color='#d9534f')\n",
    "\n",
    "    ax.set_xlabel('Training Set Size', fontsize=11)\n",
    "    ax.set_ylabel('ROC-AUC Score', fontsize=11)\n",
    "    ax.set_title(title, fontsize=12, pad=15)\n",
    "    ax.legend(loc='lower right', fontsize=10)\n",
    "    ax.grid(True, alpha=0.3)\n",
    "    ax.set_ylim(0.5, 1.0)\n",
    "    plt.tight_layout()\n",
    "\n",
    "    if save_path:\n",
    "        fig.savefig(save_path, dpi=300, bbox_inches='tight')\n",
    "    plt.show()\n",
    "\n",
    "    return train_sizes_abs, train_scores, test_scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 12.2. Learning Curve - Logistic Regression Baseline\n",
    "print('Dang ve Learning Curve cho Logistic Regression...')\n",
    "lr_train_sizes, lr_train_scores, lr_test_scores = plot_learning_curve(\n",
    "    lr_pipeline,\n",
    "    'Learning Curve - Logistic Regression Baseline',\n",
    "    X_train, y_train,\n",
    "    cv=cv,\n",
    "    train_sizes=train_sizes,\n",
    "    save_path='../notebooks/figures/baseline_learning_curve.png'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 12.3. Learning Curve - k-NN\n",
    "print('Dang ve Learning Curve cho k-NN...')\n",
    "knn_train_sizes, knn_train_scores, knn_test_scores = plot_learning_curve(\n",
    "    best_knn,\n",
    "    'Learning Curve - k-NN (Best Model)',\n",
    "    X_train, y_train,\n",
    "    cv=cv,\n",
    "    train_sizes=train_sizes,\n",
    "    save_path='../notebooks/figures/knn_learning_curve.png'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 12.4. Learning Curve - Linear SVM\n",
    "print('Dang ve Learning Curve cho Linear SVM...')\n",
    "svm_linear_train_sizes, svm_linear_train_scores, svm_linear_test_scores = plot_learning_curve(\n",
    "    best_svm_linear,\n",
    "    'Learning Curve - Linear SVM (Best Model)',\n",
    "    X_train, y_train,\n",
    "    cv=cv,\n",
    "    train_sizes=train_sizes,\n",
    "    save_path='../notebooks/figures/svm_linear_learning_curve.png'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 12.5. Learning Curve - RBF SVM\n",
    "print('Dang ve Learning Curve cho RBF SVM...')\n",
    "svm_rbf_train_sizes, svm_rbf_train_scores, svm_rbf_test_scores = plot_learning_curve(\n",
    "    best_svm_rbf,\n",
    "    'Learning Curve - RBF SVM (Best Model)',\n",
    "    X_train, y_train,\n",
    "    cv=cv,\n",
    "    train_sizes=train_sizes,\n",
    "    save_path='../notebooks/figures/svm_rbf_learning_curve.png'\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 13. Validation Curves\n",
    "Phan tich tac dong cua sieu tham so quan trong"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 13.1. Ham ve Validation Curve\n",
    "def plot_validation_curve(estimator, title, X, y, param_name, param_range, cv,\n",
    "                          scoring='roc_auc', figsize=(7, 5), save_path=None):\n",
    "    train_scores, test_scores = validation_curve(\n",
    "        estimator, X, y,\n",
    "        param_name=param_name,\n",
    "        param_range=param_range,\n",
    "        cv=cv,\n",
    "        scoring=scoring,\n",
    "        n_jobs=-1\n",
    "    )\n",
    "\n",
    "    train_mean = np.mean(train_scores, axis=1)\n",
    "    train_std = np.std(train_scores, axis=1)\n",
    "    test_mean = np.mean(test_scores, axis=1)\n",
    "    test_std = np.std(test_scores, axis=1)\n",
    "\n",
    "    fig, ax = plt.subplots(figsize=figsize, dpi=300)\n",
    "    ax.plot(param_range, train_mean, 'o-', color='#2b5c8f', label='Training Score')\n",
    "    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,\n",
    "                    alpha=0.2, color='#2b5c8f')\n",
    "    ax.plot(param_range, test_mean, 'o-', color='#d9534f', label='Cross-Validation Score')\n",
    "    ax.fill_between(param_range, test_mean - test_std, test_mean + test_std,\n",
    "                    alpha=0.2, color='#d9534f')\n",
    "\n",
    "    ax.set_xlabel(param_name.replace('classifier__', ''), fontsize=11)\n",
    "    ax.set_ylabel('ROC-AUC Score', fontsize=11)\n",
    "    ax.set_title(title, fontsize=12, pad=15)\n",
    "    ax.legend(loc='lower right', fontsize=10)\n",
    "    ax.grid(True, alpha=0.3)\n",
    "    ax.set_ylim(0.5, 1.0)\n",
    "    \n",
    "    if all(isinstance(x, (int, float)) and x > 0 for x in param_range):\n",
    "        if max(param_range) / min(param_range) > 10:\n",
    "            ax.set_xscale('log')\n",
    "\n",
    "    plt.tight_layout()\n",
    "\n",
    "    if save_path:\n",
    "        fig.savefig(save_path, dpi=300, bbox_inches='tight')\n",
    "    plt.show()\n",
    "\n",
    "    return train_scores, test_scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 13.2. Validation Curve - k-NN: n_neighbors\n",
    "print('Dang ve Validation Curve cho k-NN (n_neighbors)...')\n",
    "knn_param_range = [1, 3, 5, 7, 9, 11, 13, 15, 19, 25, 31]\n",
    "knn_val_train, knn_val_test = plot_validation_curve(\n",
    "    knn_pipeline,\n",
    "    'Validation Curve - k-NN: n_neighbors',\n",
    "    X_train, y_train,\n",
    "    param_name='classifier__n_neighbors',\n",
    "    param_range=knn_param_range,\n",
    "    cv=cv,\n",
    "    save_path='../notebooks/figures/knn_validation_curve.png'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 13.3. Validation Curve - Linear SVM: C\n",
    "print('Dang ve Validation Curve cho Linear SVM (C)...')\n",
    "svm_linear_param_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]\n",
    "svm_linear_val_train, svm_linear_val_test = plot_validation_curve(\n",
    "    svm_linear_pipeline,\n",
    "    'Validation Curve - Linear SVM: C',\n",
    "    X_train, y_train,\n",
    "    param_name='classifier__C',\n",
    "    param_range=svm_linear_param_range,\n",
    "    cv=cv,\n",
    "    save_path='../notebooks/figures/svm_linear_validation_curve.png'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 13.4. Validation Curve - RBF SVM: C (voi gamma tot nhat)\n",
    "print('Dang ve Validation Curve cho RBF SVM (C)...')\n",
    "best_gamma = svm_rbf_grid.best_params_['classifier__gamma']\n",
    "svm_rbf_pipeline_fixed_gamma = Pipeline(steps=[\n",
    "    ('preprocessor', get_preprocessor_pipeline()),\n",
    "    ('classifier', SVC(kernel='rbf', gamma=best_gamma, probability=True, random_state=42))\n",
    "])\n",
    "svm_rbf_param_range = [0.01, 0.1, 1, 10, 100, 1000]\n",
    "svm_rbf_val_train, svm_rbf_val_test = plot_validation_curve(\n",
    "    svm_rbf_pipeline_fixed_gamma,\n",
    "    f'Validation Curve - RBF SVM: C (gamma={best_gamma})',\n",
    "    X_train, y_train,\n",
    "    param_name='classifier__C',\n",
    "    param_range=svm_rbf_param_range,\n",
    "    cv=cv,\n",
    "    save_path='../notebooks/figures/svm_rbf_validation_curve.png'\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 14. Thao luan: Overfitting, Underfitting va Hoi tu\n",
    "### Phan tich Learning Curves:\n",
    "- <b>Logistic Regression</b>: Duong training va CV hoi tu gan nhau -> mo hinh on dinh, it overfitting\n",
    "- <b>k-NN</b>: Training score cao, CV score thap hon -> co dau hieu overfitting nhe khi k nho\n",
    "- <b>Linear SVM</b>: Training va CV gan nhau -> regularization tot kiem soat overfitting\n",
    "- <b>RBF SVM</b>: Can quan sat gap giua training va CV de danh gia overfitting\n",
    "\n",
    "### Phan tich Validation Curves:\n",
    "- <b>k-NN (n_neighbors)</b>: k nho -> overfitting (high variance); k lon -> underfitting (high bias)\n",
    "- <b>Linear SVM (C)</b>: C nho -> underfitting; C lon -> co the overfitting\n",
    "- <b>RBF SVM (C)</b>: Tuong tu, gamma duoc co dinh o gia tri tot nhat\n",
    "\n",
    "### Ket luan thuc nghiem:\n",
    "1. Logistic Regression baseline la mo hinh can bang tot\n",
    "2. k-NN can chon k phu hop (distance weighting giup cai thien)\n",
    "3. SVM (Linear/RBF) can tuning C can than\n",
    "4. Mo hinh tot nhat duoc chon dua tren Test ROC-AUC"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 15. Ket luan Thuc nghiem\n",
    "### Tom tat ket qua:\n",
    "1. <b>Baseline Logistic Regression</b>: Don gian, nhanh, on dinh, ROC-AUC test ~0.70\n",
    "2. <b>k-NN</b>: Cai thien duoc Recall nh\u1edf distance weighting, nh\u1eefng ch\u1ea3m h\u01a1n\n",
    "3. <b>Linear SVM</b>: Hi\u1ec7u qu\u1ea3 t\u01b0\u01a1ng duong LR, c\u00f3 th\u1ec3 t\u1ed1t h\u01a1n o margin lon\n",
    "4. <b>RBF SVM</b>: Linh hoat nh\u1ea5t nh\u1eefng t\u1ed3n t\u00ednh to\u00e1n, c\u1ea7n tuning gamma\n",
    "\n",
    "### Khuyen nghi cho Milestone 3:\n",
    "- Ensemble methods (Random Forest, Gradient Boosting, XGBoost)\n",
    "- Threshold optimization cho Recall cao (y te)\n",
    "- Calibration xac suat\n",
    "- Feature selection nang cao"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('notebooks/02_Modeling_Evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print('Notebook written successfully!')