## Dashboard / Model Evaluation Metrics

### Precision
- Measures how many predicted failures were actually failures.
- High precision means the model makes few false alarms.

### Recall
- Measures how many actual failures the model detected.
- High recall means the model misses fewer real failures.

### F1 Score
- The harmonic mean of precision and recall.
- Balances false alarms and missed failures into one single score.

### PR-AUC (Precision-Recall AUC)
- The area under the precision-recall curve.
- Shows overall classifier quality on imbalanced data, especially for rare failure detection.

### Sensor Telemetry
- Live sensor measurements collected from the machine.
- Examples: air temperature, process temperature, rotational speed, torque, tool wear.
- These values are the input features the model uses to predict failures.

### Failure Distribution
- Shows how many failure vs normal events exist in the dataset.
- Helps understand class balance and how rare failures are.

### Risk Distribution
- Breaks predicted failure probabilities into risk levels: LOW / MEDIUM / HIGH / CRITICAL.
- Indicates how many cases fall into each risk category based on the model’s output.

These metrics tell you:
- model performance on historical/test data (`Precision`, `Recall`, `F1`, `PR-AUC`),
- what the sensors are reporting (`Sensor Telemetry`),
- how common failures are in the dataset (`Failure Distribution`),
- and how the model ranks severity for predicted events (`Risk Distribution`).