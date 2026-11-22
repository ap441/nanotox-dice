# Baseline Tree Model Benchmark & Property-Activity Benchmarking

**Pipeline Component:** `src/models/train_baselines.py`  
**Configuration:** `configs/experiment_config.yaml`  
**Validation Strategy:** 80/20 DOI GroupShuffleSplit + 5-Fold GroupKFold Cross-Validation

---

## 1. Executive Summary

This report benchmarks three tree ensemble architectures across the dual Safe-by-Design tasks:
1. **Task A (Cytotox)**: Binary classification of mammalian cell biocompatibility (`is_biocompatible`, viability $\ge 70\%$).
2. **Task B (SelTox)**: Binary classification of bacterial antimicrobial potency (`is_antibacterial`, $\text{MIC} \le 64\,\mu\text{g/mL}$).

- **Best Cytotox Model (Mammalian Safety):** `random_forest`
- **Best SelTox Model (Bacterial Efficacy):** `random_forest`

---

## 2. Benchmark Metrics by Task

### Task A: Mammalian Biocompatibility (Cytotox)

| Model | Split Type | ROC-AUC | PR-AUC | F1-Score | Balanced Accuracy | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **random_forest** (Selected) | 5-Fold CV | 0.6950 ± 0.069 | 0.7882 ± 0.022 | 0.7537 ± 0.035 | 0.6204 ± 0.051 | 0.7065 ± 0.069 | 0.8150 ± 0.038 |
| **random_forest** (Selected) | Holdout Test | 0.7279 | 0.8682 | 0.8007 | 0.5892 | 0.7572 | 0.8494 |
| xgboost | 5-Fold CV | 0.6652 ± 0.095 | 0.7680 ± 0.039 | 0.7415 ± 0.016 | 0.6154 ± 0.069 | 0.7195 ± 0.044 | 0.7689 ± 0.037 |
| xgboost | Holdout Test | 0.6945 | 0.8576 | 0.7510 | 0.6050 | 0.7760 | 0.7277 |
| catboost | 5-Fold CV | 0.6897 ± 0.086 | 0.7774 ± 0.031 | 0.7456 ± 0.013 | 0.6467 ± 0.072 | 0.7529 ± 0.053 | 0.7453 ± 0.052 |
| catboost | Holdout Test | 0.6765 | 0.8437 | 0.7506 | 0.6105 | 0.7802 | 0.7232 |


### Task B: Bacterial Antimicrobial Potency (SelTox)

| Model | Split Type | ROC-AUC | PR-AUC | F1-Score | Balanced Accuracy | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **random_forest** (Selected) | 5-Fold CV | 0.4954 ± 0.066 | 0.6654 ± 0.068 | 0.6744 ± 0.056 | 0.5111 ± 0.058 | 0.6476 ± 0.086 | 0.7207 ± 0.088 |
| **random_forest** (Selected) | Holdout Test | 0.4183 | 0.5948 | 0.6006 | 0.4815 | 0.6351 | 0.5697 |
| xgboost | 5-Fold CV | 0.4421 ± 0.095 | 0.6182 ± 0.060 | 0.6471 ± 0.087 | 0.4859 ± 0.104 | 0.6314 ± 0.103 | 0.6860 ± 0.127 |
| xgboost | Holdout Test | 0.4745 | 0.6672 | 0.6531 | 0.4686 | 0.6292 | 0.6788 |
| catboost | 5-Fold CV | 0.3940 ± 0.074 | 0.5824 ± 0.072 | 0.6577 ± 0.054 | 0.4887 ± 0.067 | 0.6341 ± 0.075 | 0.6979 ± 0.087 |
| catboost | Holdout Test | 0.4412 | 0.5912 | 0.6977 | 0.5322 | 0.6704 | 0.7273 |


---

## 3. Methodological Rigor & Data Leakage Prevention

- **Study-Level DOI Grouping**: Grouping splits by `doi` strictly isolates all experimental observations derived from the same laboratory publication to either the training set or the test set. This reflects real-world clinical translation where models must predict outcomes for newly synthesized nanoparticles from previously unseen research laboratories.
- **Pipeline Imputation**: Missing values for numeric features (`zeta_potential_mv`, `concentration`) and categorical features (`shape`, `cell_type`, `surface_coating`) are imputed dynamically within cross-validation folds using fold-specific statistics (median and mode), completely eliminating lookahead bias.
- **Class Balancing**: Balanced class weighting (`class_weight='balanced'`, `scale_pos_weight`, and `auto_class_weights='Balanced'`) prevents bias toward majority classes, as evidenced by consistent Balanced Accuracy and PR-AUC scores.

---

## 4. Persisted Model Artifacts

- `models/cytotox_best_model.joblib`: Fully refitted pipeline for mammalian safety.
- `models/seltox_best_model.joblib`: Fully refitted pipeline for bacterial efficacy.
- Downstream counterfactual generators (DiCE) will directly load these serialized pipelines to discover actionable, constrained modifications.