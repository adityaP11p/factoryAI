# Architecture Overview

System Flow :

Raw Sensor Data
        ↓
Feature Engineering
        ↓
Feature Dataset (features.csv)
        ↓
Model Training (XGBoost)
        ↓
Saved Pipeline (pipeline.pkl)
        ↓
Flask API (/predict)
        ↓
Prediction Output
        ↓
SHAP Explanation

---

Components :

1. Data Layer

- Input: Sensor data
- Output: Engineered features

2. Feature Engineering

- Rolling statistics
- Lag features
- Derived features

3. Model Layer

- XGBoost classifier
- Handles imbalance

4. API Layer

- Flask server
- Accepts JSON input
- Returns prediction

5. Explainability Layer

- SHAP
- Provides feature-level explanations

---

Key Design Decisions :

- PR-AUC used due to imbalance
- Pipeline used to ensure consistent preprocessing
- SHAP added for interpretability