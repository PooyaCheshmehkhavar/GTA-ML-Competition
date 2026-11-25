# GTA Sales Prediction: Two-Stage Hybrid ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Optimized-yellow.svg)](https://catboost.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced competitive machine learning solution implementing a **hierarchical two-stage classification and regression pipeline** to predict digital/physical sales channels and unit sales.

---

## 🎯 Problem Overview & Architecture

Predicting video game retail velocity requires modeling both discrete channel preference and continuous sales volume under heavy right-skewed distributions. 

Rather than treating classification and regression independently, this pipeline implements **Meta-Feature Stacking**:

```
[Raw Features] 
       │
       ▼
┌──────────────────────────────────────────────┐
│ Stage 1: Sales Channel Classification        │
│ • CatBoostClassifier + LightGBMClassifier    │
│ • Blended Probability: P(Sales Channel = Dig)│
└──────────────────────┬───────────────────────┘
                       │ Meta-Feature (prob_digital)
                       ▼
┌──────────────────────────────────────────────┐
│ Stage 2: Units Sold Regression               │
│ • XGBRegressor + LGBMRegressor + CatBoost    │
│ • Loss Metric: Mean Squared Log Error (MSLE) │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
            [Final Submission]
```

---

## 🛠 Feature Engineering & Modeling Strategy

1. **Preprocessing & Encoding:**
   - Categorical feature encoding via Scikit-Learn's `OrdinalEncoder`.
   - Handling skewed targets with log transformations.
2. **Stage 1 (Classification):**
   - Ensemble of `LGBMClassifier` (1,200 estimators, depth=6, lr=0.03) and `CatBoostClassifier`.
   - Probability blending: P_ensemble = 0.5 * P_LGBM + 0.5 * P_CatBoost.
3. **Stage 2 (Regression):**
   - Incorporating predicted channel probability `prob_digital` as an interaction feature.
   - Ensembling Gradient Boosted Decision Trees (GBDTs) across LightGBM, CatBoost, and XGBoost.
   - Target metric: **Mean Squared Logarithmic Error (MSLE)**.

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PooyaCheshmehkhavar/GTA-ML-Competition.git
   cd GTA-ML-Competition
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn lightgbm xgboost catboost jupyter
   ```

3. **Launch the Notebook:**
   ```bash
   jupyter notebook GTA.ipynb
   ```

---

## 👤 Author
**Pooya Cheshmehkhavar**  
- GitHub: [@PooyaCheshmehkhavar](https://github.com/PooyaCheshmehkhavar)
