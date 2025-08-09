# Machine Learning Classification Project — COMPAS & Diabetes Datasets

## 📌 Overview
This project implements machine learning techniques to solve binary classification problems using two datasets:

- **COMPAS**: Predicting recidivism within two years.
- **Diabetes**: Predicting the onset of diabetes.

We explore **data analysis, preprocessing, feature engineering, model selection**, and **hyperparameter optimization**, and evaluate results on development and test sets.

---

## 📊 Datasets

### 1. COMPAS
- **Train size**: 5,049 samples  
- **Dev size**: 721 samples  
- **Features**: 8 categorical features (`sex`, `age`, `race`, `juv_fel_count`, `juv_misd_count`, `juv_other_count`, `priors_count`, `c_charge_degree`)  
- **Target**: `two_year_recid` (binary)  
- **Class distribution**: 54.49% (label 0), 45.51% (label 1)  
- **Challenge**: Low correlation between features and target (max = 0.27).

### 2. Diabetes
- **Train size**: 537 samples  
- **Dev size**: 77 samples  
- **Features**: 8 numeric features (`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`)  
- **Target**: `Outcome` (binary)  
- **Class distribution**: 64.98% (label 0), 35.02% (label 1)  
- **Advantage**: Higher correlation between features and target compared to COMPAS.

---

## 🛠 Data Preprocessing
- **Missing values**:
  - Numeric: Filled with median (via `SimpleImputer`).
  - Categorical: Filled with most frequent value.
  - For Diabetes: Zero values in `BloodPressure`, `BMI`, `Glucose`, `Insulin`, `SkinThickness` treated as missing and replaced with median.
- **Outliers**: Removed based on z-score (train set only).
- **Scaling**: Standardization with `StandardScaler`.
- **Balancing**: Applied **SMOTE** to address class imbalance.

---

## 🔬 Feature Engineering
- **Decomposition & selection**:
  - Factor Analysis
  - Mutual Information
  - Variance Threshold
- **New features**:
  - Ratio features (e.g., `Glucose_to_BMI` in Diabetes).
  - Threshold-based categorical features for `Glucose`, `BloodPressure`, `BMI` (Diabetes only).
- **Transformations**:
  - `QuantileTransformer`: Normalize distribution closer to Gaussian.
  - `KBinsDiscretizer`: Discretize numeric features into categorical bins.

---

## 🤖 Models Used
- **Gradient Boosting** — ensemble of weak learners for better performance.
- **CatBoost** — optimized for categorical features.
- **XGBoost** — improved gradient boosting with speed and efficiency.

---

## ⚙ Hyperparameter Optimization
- Used **Optuna** to optimize model hyperparameters based on **F1-score** on the development set.

---

## 📈 Results

| Dataset  | Set   | F1-score | Accuracy |
|----------|-------|----------|----------|
| COMPAS   | Dev   | 69.06%   | 69.35%   |
| COMPAS   | Test  | 67.76%   | 68.01%   |
| Diabetes | Dev   | 85.70%   | 85.71%   |
| Diabetes | Test  | 73.72%   | 75.32%   |

---

## 📝 Conclusion
- **COMPAS**: Achieved stable generalization with minimal gap between dev and test performance.
- **Diabetes**: Significant performance drop from dev to test due to:
  - Possible overfitting on dev set.
  - Missing value handling strategy not generalizing to unseen data.
  - Distribution shift between dev and test sets.

---

## 📂 Repository Structure
