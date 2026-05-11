# FactoryGuard AI – IoT Predictive Maintenance Engine

```bash
# Execute Full 7-Stage End-To-End Training (Ingest -> Tune -> Train -> SHAP)
python src/run_pipeline.py

# Alternatively, skip the lengthy Optuna Tuning phase using Defaults
python src/run_pipeline.py --skip-tuning

# Granular debugging execution allows for isolated stage commands:
python src/run_pipeline.py --stage ingest
python src/run_pipeline.py --stage train
```

FactoryGuard AI is a machine learning system designed to predict industrial machine failures using sensor data. The system processes time-series telemetry such as temperature, rotational speed, torque, and tool wear, and predicts failure probability using a trained model.

---

Project Overview :

This project focuses on:

- Feature engineering from sensor data
- Training a predictive model for failure detection
- Deploying the model using a Flask API
- Explaining predictions using SHAP (Explainable AI)

---

Features :

- Feature Engineering
  
  - Rolling mean (window = 6)
  - Lag features (t-1, t-2)
  - Temperature difference ("temp_diff")
  - Interaction feature ("wear_torque")

- Model
  
  - XGBoost classifier
  - Handles class imbalance
  - Evaluated using PR-AUC

- API Deployment
  
  - Flask-based REST API
  - "/predict" endpoint for real-time predictions

- Explainability
  
  - SHAP waterfall plot (single prediction)
  - SHAP summary plot (global importance)

---

Project Structure :

data/
 ├── features/
 │    └── features.csv

src/
 ├── train.py
 ├── predict.py
 ├── explain.py
 ├── config.py

models/
 ├── pipeline.pkl
 ├── shap_waterfall.png
 ├── shap_summary.png

app.py
ARCHITECTURE.md
FEATURE_IMPLEMENTATION.md
implementation_plan.md
requirements.txt
README.md

---

Model Performance :

- Metric: PR-AUC
- Score: ~0.75

This metric is used because the dataset is highly imbalanced.

---

Running the Project :

1. Install dependencies

pip install python==3.10.9

pip install -r requirements.txt

2. Train the model

python -m src.train

3. Run API

python app.py

4. Test API

Send POST request to:

http://127.0.0.1:5000/predict

---

Example Input :

{
  "air_temp": 300,
  "process_temp": 310,
  "rotational_speed": 1500,
  "torque": 40,
  "tool_wear": 5,
  "Type_L": 1,
  "Type_M": 0,
  "air_temp_mean_6": 299,
  "process_temp_mean_6": 308,
  "rpm_std_6": 20,
  "torque_std_6": 5,
  "air_temp_lag1": 298,
  "air_temp_lag2": 297,
  "temp_diff": 10,
  "wear_torque": 200
}

---

SHAP Explainability :

Waterfall Plot

"Waterfall" (models/shap_waterfall.png)

Summary Plot

"Summary" (models/shap_summary.png)

Interpretation

- "wear_torque", "torque", and "tool_wear" are the most important features
- These features increase failure probability significantly
- Temperature features have smaller influence

---

Conclusion :

The system successfully predicts machine failure using engineered features and provides interpretable results using SHAP.

---

Future Improvements :

- Add multiple models (ensemble)
- Hyperparameter tuning
- Real-time streaming data integration
