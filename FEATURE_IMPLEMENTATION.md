# Feature Engineering Implementation

Overview  :

Feature engineering transforms raw sensor data into meaningful inputs for the model.

---

Features Created :

1. Rolling Mean

air_temp_mean_6
process_temp_mean_6

- Captures short-term trends
- Smooths noise

---

2. Rolling Standard Deviation

rpm_std_6
torque_std_6

- Measures variability
- Detects instability

---

3. Lag Features

air_temp_lag1
air_temp_lag2

- Captures temporal dependency
- Uses previous values

---

4. Temperature Difference

temp_diff = process_temp - air_temp

- Indicates thermal stress

---

5. Interaction Feature

wear_torque = tool_wear × torque

- Combines wear and load
- Strong predictor of failure

---

Importance :

- Improves model performance
- Captures time-series behavior
- Introduces domain knowledge

---

Conclusion :

Feature engineering is the most critical part of this project, enabling the model to detect patterns related to machine failure.