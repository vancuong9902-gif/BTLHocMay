import re
p='03_SourceCode/run_svm_experiments.py'
s=open(p,encoding='utf-8').read()
s=s.replace("import os\nimport sys\n","import os\nimport sys\nfrom pathlib import Path\n")
s=s.replace("sys.path.append(os.path.abspath('..'))","sys.path.insert(0, str(Path(__file__).resolve().parent))")
s=s.replace("data_path = '../data/framingham.csv'","data_path = str(Path(__file__).resolve().parent / 'data' / 'framingham.csv')")
# path fixes
for a,b in [
 ("svm_linear_results_df.to_csv('../models/results/svm_linear_gridsearch_results.csv', index=False)","svm_linear_results_df.to_csv(str(Path(__file__).resolve().parent / 'models' / 'results' / 'svm_linear_gridsearch_results.csv'), index=False)"),
 ("joblib.dump(best_svm_linear, '../models/best_svm_linear.joblib')","joblib.dump(best_svm_linear, str(Path(__file__).resolve().parent / 'models' / 'best_svm_linear.joblib'))"),
 ("pd.DataFrame([svm_linear_test_metrics]).to_csv('../models/results/svm_linear_test_metrics.csv', index=False)","pd.DataFrame([svm_linear_test_metrics]).to_csv(str(Path(__file__).resolve().parent / 'models' / 'results' / 'svm_linear_test_metrics.csv'), index=False)"),
 ("pd.DataFrame([svm_linear_grid.best_params_]).to_csv('../models/results/svm_linear_best_params.csv', index=False)","pd.DataFrame([svm_linear_grid.best_params_]).to_csv(str(Path(__file__).resolve().parent / 'models' / 'results' / 'svm_linear_best_params.csv'), index=False)"),
 ("svm_rbf_results_df.to_csv('../models/results/svm_rbf_gridsearch_results.csv', index=False)","svm_rbf_results_df.to_csv(str(Path(__file__).resolve().parent / 'models' / 'results' / 'svm_rbf_gridsearch_results.csv'), index=False)"),
 ("joblib.dump(best_svm_rbf, '../models/best_svm_rbf.joblib')","joblib.dump(best_svm_rbf, str(Path(__file__).resolve().parent / 'models' / 'best_svm_rbf.joblib'))"),
 ("pd.DataFrame([svm_rbf_test_metrics]).to_csv('../models/results/svm_rbf_test_metrics.csv', index=False)","pd.DataFrame([svm_rbf_test_metrics]).to_csv(str(Path(__file__).resolve().parent / 'models' / 'results' / 'svm_rbf_test_metrics.csv'), index=False)"),
 ("pd.DataFrame([svm_rbf_grid.best_params_]).to_csv('../models/results/svm_rbf_best_params.csv', index=False)","pd.DataFrame([svm_rbf_grid.best_params_]).to_csv(str(Path(__file__).resolve().parent / 'models' / 'results' / 'svm_rbf_best_params.csv'), index=False)"),
]:
    s=s.replace(a,b)
s=s.replace("param_grid_svm_linear = {'classifier__C': [0.01, 0.1, 1, 10, 100]}","param_grid_svm_linear = {'classifier__C': [0.01, 0.1, 1, 10, 100], 'classifier__class_weight': [None, 'balanced']}")
s=s.replace("""param_grid_svm_rbf = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 0.01, 0.1, 1]
}""","""param_grid_svm_rbf = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 0.01, 0.1, 1],
    'classifier__class_weight': [None, 'balanced']
}""")
open(p,'w',encoding='utf-8').write(s)
print("done")
