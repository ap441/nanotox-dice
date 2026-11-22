# Nanotoxicology Preprocessing & Cohort Harmonization Report

**Pipeline Component:** `src/data/preprocess.py`  
**Configuration:** `configs/experiment_config.yaml`

---

## 1. Attrition Summary

| Dataset | Initial Rows | Dropped / Filtered | Final Clean Rows | Retention % | Primary Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cytotox (Mammalian)** | 5,476 | 56 | 5,420 | 98.98% | `viability_%`, `is_biocompatible` |
| **SelTox (Bacterial)** | 3,244 | 1,317 | 1,927 | 59.40% | `mic_ug_ml`, `log10_mic`, `is_antibacterial` |


### Attrition Breakdown Details

- **Cytotox Filtering**:
  - Dropped null `viability_%`: 13 rows
  - Dropped null `particle_size_nm`: 43 rows
  - Net rows dropped: 56 rows

- **SelTox Filtering**:
  - Filtered out non-MIC/MBC or null assay rows: 1259 rows
  - Filtered out non-numeric/bounded MIC strings (11): ['>150', '>250', '>100', '>100', '>100', '>100', '>100', '>100', '>100', '>100', '>5000']
  - Filtered out non-positive ($\le 0$) MIC values: 47 rows
  - Net valid assay rows retained: 1,927 rows

---

## 2. Target Distributions & Class Balance

### Cytotox: Host Cell Safety
- **Biocompatible (`viability_% >= 70.0%`)**: 3,526 (65.06%)
- **Cytotoxic (`viability_% < 70.0%`)**: 1,894 (34.94%)
- **Viability Clipping**: Clipped 106 rows with viability $> 120.0\%$ down to 120.0% (assay noise/hormesis). 0 rows $< 0.0\%$.

### SelTox: Antimicrobial Efficacy
- **Antibacterial Active (`MIC <= 64.0 µg/mL`)**: 1,237 (64.19%)
- **Antibacterial Inactive / Resistant (`MIC > 64.0 µg/mL`)**: 690 (35.81%)
- **log10(MIC) Quartiles**: Min=-2.40, 25%=0.80, Median=1.51, 75%=2.10, Max=4.10

---

## 3. Feature Missingness in Cleaned Datasets

> [!NOTE]
> Missing numeric features (such as `zeta_potential_mv` and `concentration`) are intentionally **not imputed** globally here. Imputation is deferred to cross-validated model pipelines to avoid train-test data leakage.

### Cytotox Clean Features Missingness

| Column | Non-Null Count | Null Count | Null % | Data Role |
| :--- | :--- | :--- | :--- | :--- |
| `core_material` | 5,420 | 0 | 0.00% | Feature |
| `particle_size_nm` | 5,420 | 0 | 0.00% | Feature |
| `zeta_potential_mv` | 3,174 | 2,246 | 41.44% | Feature |
| `surface_coating` | 5,420 | 0 | 0.00% | Feature |
| `is_coated` | 5,420 | 0 | 0.00% | Flag |
| `shape` | 5,420 | 0 | 0.00% | Feature |
| `cell_type` | 5,420 | 0 | 0.00% | Feature |
| `concentration` | 3,123 | 2,297 | 42.38% | Feature |
| `time_hr` | 5,420 | 0 | 0.00% | Feature |
| `test_assay` | 5,373 | 47 | 0.87% | Feature |
| `doi` | 5,420 | 0 | 0.00% | Feature |
| `viability_%` | 5,420 | 0 | 0.00% | Target |
| `is_biocompatible` | 5,420 | 0 | 0.00% | Target |
| `is_shared_material` | 5,420 | 0 | 0.00% | Flag |

### SelTox Clean Features Missingness

| Column | Non-Null Count | Null Count | Null % | Data Role |
| :--- | :--- | :--- | :--- | :--- |
| `core_material` | 1,927 | 0 | 0.00% | Feature |
| `particle_size_nm` | 1,842 | 85 | 4.41% | Feature |
| `zeta_potential_mv` | 618 | 1,309 | 67.93% | Feature |
| `is_coated` | 1,927 | 0 | 0.00% | Flag |
| `shape` | 1,927 | 0 | 0.00% | Feature |
| `bacteria` | 1,927 | 0 | 0.00% | Feature |
| `mdr` | 1,927 | 0 | 0.00% | Feature |
| `time_hr` | 1,770 | 157 | 8.15% | Feature |
| `method` | 1,927 | 0 | 0.00% | Feature |
| `doi` | 1,927 | 0 | 0.00% | Feature |
| `mic_ug_ml` | 1,927 | 0 | 0.00% | Target |
| `log10_mic` | 1,927 | 0 | 0.00% | Target |
| `is_antibacterial` | 1,927 | 0 | 0.00% | Target |
| `is_shared_material` | 1,927 | 0 | 0.00% | Flag |

---

## 4. Shared Core Materials Coverage

- **Cytotox Records with Shared Material**: 3,971 / 5,420 (73.27%)
- **SelTox Records with Shared Material**: 1,912 / 1,927 (99.22%)

### Shape Distribution Across Both Datasets

| Canonical Shape | Cytotox Count (Pct) | SelTox Count (Pct) |
| :--- | :--- | :--- |
| `spherical` | 2,551 (47.1%) | 1,410 (73.2%) |
| `rod` | 182 (3.4%) | 95 (4.9%) |
| `nanotube` | 223 (4.1%) | 12 (0.6%) |
| `cubic` | 16 (0.3%) | 67 (3.5%) |
| `other` | 206 (3.8%) | 54 (2.8%) |
| `unknown` | 2,242 (41.4%) | 289 (15.0%) |