# FactoryGuard AI – IoT Predictive Maintenance Engine

An intelligent, full-stack predictive maintenance system that uses machine learning to detect machine failures before they happen. Features real-time sensor data ingestion, persistent database storage, explainable AI predictions, and an interactive dashboard.

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize the database
python -c "from src.database import init_db; db = init_db('models/factory_guard.db'); print('✅ Database ready!')"
```

### Run the Application

```bash
# train the model
python src/run_pipeline.py
# Start the Flask web application
python app.py

# Open in browser
# http://127.0.0.1:5000
```

### Send Live Sensor Data

```bash
# Example: Send real-time sensor readings from an IoT device
python live_sensor_example.py

# Or use cURL to send data directly
curl -X POST http://localhost:5000/api/live-sensor-data \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "machine_7845",
    "air_temp": 42.5,
    "process_temp": 65.3,
    "rotational_speed": 3200,
    "torque": 68.5,
    "tool_wear": 215,
    "source": "iot_gateway"
  }'
```

---

## 📊 Project Overview

**FactoryGuard AI** is a machine learning system designed to predict industrial machine failures using sensor data. The system processes time-series telemetry (temperature, rotational speed, torque, tool wear) to identify machines at risk of failure before maintenance is needed.

### Key Features

- **Real-Time Predictions**: ML-powered failure detection with probability scores
- **Persistent Storage**: SQLite database for all predictions, sensor data, and resolutions
- **Explainability**: SHAP-based feature importance and waterfall plots
- **Live Sensor Ingestion**: Direct API endpoint for IoT devices and sensors
- **Ensemble Models**: Multiple models for consensus voting and confidence scores
- **Interactive Dashboard**: Real-time visualization of machine health and predictions
- **Audit Trail**: Track who resolved issues and when

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                   FactoryGuard AI                       │
├─────────────────────────────────────────────────────────┤
│  Frontend (Templates)                                   │
│  ├── index.html - Interactive dashboard                │
│  └── Real-time data visualization                      │
├─────────────────────────────────────────────────────────┤
│  Backend (Flask API)                                    │
│  ├── /api/predict - Manual predictions                 │
│  ├── /api/simulate - Simulated live data               │
│  ├── /api/machines-at-risk - Active failures           │
│  ├── /api/live-sensor-data - Real-time ingestion       │
│  ├── /api/resolve-machine - Mark failure as resolved   │
│  ├── /api/dashboard-data - All metrics                 │
│  └── /api/database-stats - System health               │
├─────────────────────────────────────────────────────────┤
│  ML Layer                                               │
│  ├── XGBoost, LightGBM, Logistic Regression, RF        │
│  ├── Ensemble consensus voting                         │
│  ├── SHAP explainability                               │
│  └── Feature engineering & scaling                     │
├─────────────────────────────────────────────────────────┤
│  Persistence Layer                                      │
│  ├── SQLite Database (factory_guard.db)                │
│  ├── Machine failures table                            │
│  ├── Resolved machines registry                        │
│  ├── Live sensor history                               │
│  └── Machine registry                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
factoryAI/
├── app.py                              # Flask application & API routes
├── requirements.txt                    # Python dependencies
│
├── src/
│   ├── config.py                       # Configuration & paths
│   ├── database.py                     # SQLite database manager (NEW)
│   ├── prediction_logger.py            # Prediction history logging
│   ├── factory_health.py               # Factory health monitoring
│   ├── explainability.py               # SHAP explanation
│   └── [other modules]
│
├── models/
│   ├── factory_guard.db                # SQLite database (auto-created)
│   ├── xgboost_model.joblib            # Trained XGBoost model
│   ├── lightgbm_model.joblib           # Trained LightGBM model
│   ├── random_forest_model.joblib      # Trained RF model
│   ├── logistic_regression_model.joblib # Trained LR model
│   ├── pipeline.pkl                    # Preprocessing pipeline
│   ├── scaler.pkl                      # Feature scaler
│   ├── feature_importance.csv          # Feature rankings
│   ├── shap_waterfall.png              # SHAP visualization
│   └── shap_summary.png                # SHAP summary plot
│
├── data/
│   ├── features/
│   │   └── features.csv                # Engineered features dataset
│   └── [other data files]
│
├── templates/
│   └── index.html                      # Web dashboard UI
│
├── static/
│   ├── js/
│   │   ├── dashboard.js                # Frontend logic
│   │   └── [other scripts]
│   └── css/
│       └── [stylesheets]
│
├── notebooks/                          # Analysis & exploration
│
├── IMPLEMENTATION_SUMMARY.md           # Recent changes & features
├── SQLITE_DATABASE_GUIDE.md            # Database documentation
├── TECHNICAL_DETAILS.md                # Architecture & internals
├── ARCHITECTURE.md                     # System design
├── FEATURE_IMPLEMENTATION.md           # Feature engineering details
├── live_sensor_example.py              # Example: Live data ingestion
└── test_sqlite_implementation.py       # Database tests

```

---

## 🔧 Feature Engineering

The system uses advanced feature engineering from raw sensor data:

- **Rolling Statistics**: Mean and std deviation over 6-step windows
- **Lag Features**: Previous timestep values (t-1, t-2)
- **Derived Features**:
  - `temp_diff`: Difference between process and air temperature
  - `wear_torque`: Interaction between tool wear and torque
- **Type Encoding**: One-hot encoding for machine types (L, M, H)

These engineered features are optimized for the XGBoost classifier and class imbalance.

---

## 🤖 Machine Learning Models

### Primary Model: XGBoost

- **Type**: Gradient Boosting Classifier
- **Metric**: PR-AUC (Precision-Recall Area Under Curve)
- **Performance**: ~0.75 PR-AUC on test set
- **Reason for PR-AUC**: Dataset is highly imbalanced (failures are rare)

### Ensemble Approach

For confidence and consensus voting, the system includes:

- XGBoost
- LightGBM
- Random Forest
- Logistic Regression

Ensemble agreement score indicates model consensus.

---

## 📡 API Endpoints

### Prediction Endpoints

#### `POST /api/predict`
Make a single prediction with sensor data.

```json
{
  "machine_id": "machine_001",
  "air_temp": 300,
  "process_temp": 310,
  "rotational_speed": 1500,
  "torque": 40,
  "tool_wear": 5,
  "Type_L": 1,
  "Type_M": 0
}
```

**Response:**
```json
{
  "prediction": 0,
  "failure_probability": 0.15,
  "risk_level": "LOW",
  "ensemble_agreement": 1.0
}
```

#### `POST /api/live-sensor-data` ⭐ NEW
Ingest real-time sensor data from IoT devices. Automatically makes prediction and logs to database.

```json
{
  "machine_id": "machine_7845",
  "air_temp": 42.5,
  "process_temp": 65.3,
  "rotational_speed": 3200,
  "torque": 68.5,
  "tool_wear": 215,
  "source": "iot_gateway"
}
```

---

### Dashboard & Analytics Endpoints

#### `GET /api/dashboard-data`
Comprehensive dashboard metrics (model performance, sensor data, risk distribution)

#### `GET /api/machines-at-risk`
Get all unresolved machine failures from database with SHAP explanations

**Response:**
```json
{
  "machines": [
    {
      "machine_id": "machine_7845",
      "risk_level": "CRITICAL",
      "avg_risk_score": 0.92,
      "recent_readings": {
        "air_temp": 42.5,
        "process_temp": 65.3
      },
      "explanation": [
        { "feature": "wear_torque", "impact": 0.45 },
        { "feature": "tool_wear", "impact": 0.38 }
      ]
    }
  ]
}
```

#### `POST /api/resolve-machine` ⭐ NEW
Permanently mark a machine failure as resolved (persists in database).

```json
{
  "machine_id": "machine_7845",
  "resolved_by": "maintenance_team",
  "notes": "Completed tool replacement"
}
```

#### `GET /api/database-stats` ⭐ NEW
Check database health and statistics.

#### `GET /api/simulation`
Generate simulated sensor data for testing and demo purposes.

#### `GET /api/factory-health`
Overall factory health metrics and aggregate risk.

---

## 🔍 Explainability

### SHAP Integration

The system uses SHAP (SHapley Additive exPlanations) to explain individual predictions:

**Features:**
- Waterfall plots for individual predictions
- Summary plots for global feature importance
- Top 3 feature impacts displayed per machine

**Output Format:**
```json
{
  "feature": "wear_torque",
  "impact": 0.45
}
```

---

## 💾 Database Schema

### Machine Failures Table
```sql
CREATE TABLE machine_failures (
  id INTEGER PRIMARY KEY,
  timestamp DATETIME,
  machine_id TEXT NOT NULL,
  prediction INTEGER,
  failure_probability REAL,
  risk_level TEXT,
  air_temp REAL,
  process_temp REAL,
  rotational_speed REAL,
  torque REAL,
  tool_wear REAL,
  resolved INTEGER DEFAULT 0,
  resolved_timestamp DATETIME,
  notes TEXT
);
```

### Resolved Machines Registry
```sql
CREATE TABLE resolved_machines (
  machine_id TEXT PRIMARY KEY,
  resolved_timestamp DATETIME,
  resolved_by TEXT,
  notes TEXT
);
```

### Live Sensor Data
```sql
CREATE TABLE live_sensor_data (
  id INTEGER PRIMARY KEY,
  timestamp DATETIME,
  machine_id TEXT NOT NULL,
  air_temp REAL,
  process_temp REAL,
  rotational_speed REAL,
  torque REAL,
  tool_wear REAL,
  source TEXT,
  metadata TEXT
);
```

### Machine Registry
```sql
CREATE TABLE machine_registry (
  machine_id TEXT PRIMARY KEY,
  created_timestamp DATETIME,
  last_seen DATETIME,
  status TEXT,
  location TEXT,
  metadata TEXT
);
```

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **PR-AUC** | ~0.75 |
| **Precision** | High |
| **Recall** | Optimized for Recall (catch failures) |
| **Accuracy** | ~95% (test set) |
| **Ensemble Agreement** | 90%+ |

> **Note**: PR-AUC is the primary metric due to severe class imbalance (failures are rare events).

---

## 🔌 Integration Examples

### IoT Gateway Integration

```python
import requests

def send_machine_status(machine_id, sensors):
    response = requests.post('http://localhost:5000/api/live-sensor-data', 
                            json={**sensors, 'machine_id': machine_id})
    return response.json()

# Send real-time data
send_machine_status('machine_7845', {
    'air_temp': 42.5,
    'process_temp': 65.3,
    'rotational_speed': 3200,
    'torque': 68.5,
    'tool_wear': 215
})
```

### MQTT Broker Integration

```python
import paho.mqtt.client as mqtt
import json
import requests

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload)
    requests.post('http://localhost:5000/api/live-sensor-data', json=payload)

client = mqtt.Client()
client.on_message = on_message
client.connect("mqtt.example.com", 1883)
client.subscribe("factory/machines/+/sensors")
client.loop_forever()
```

---

## 🧪 Testing & Verification

### Verify Database Setup
```bash
ls -la models/factory_guard.db
```

### Query Database Directly
```bash
sqlite3 models/factory_guard.db "SELECT * FROM machine_failures LIMIT 5;"
```

### Test Live Data Ingestion
```bash
python live_sensor_example.py
```

### Check System Health
```bash
curl http://localhost:5000/api/database-stats
```

---

## ✅ Key Improvements (Recent)

| Issue | Before | After |
|-------|--------|-------|
| **Delete "Machines at Risk"** | Temporary (localStorage) | ✅ Permanent (SQLite) |
| **Persistence** | Lost on page refresh | ✅ Survives everything |
| **Live Sensor Ingestion** | No endpoint | ✅ POST /api/live-sensor-data |
| **Real-Time Failures** | Simulated only | ✅ From IoT devices |
| **Audit Trail** | None | ✅ Who resolved, when, why |
| **Data Retention** | CSVs only | ✅ Queryable SQLite DB |
| **Machine Tracking** | Manual | ✅ Auto-registration |
| **Historical Analysis** | Limited | ✅ Full sensor history |

---

## 🎯 Next Steps

1. **Connect IoT Devices**: Use `/api/live-sensor-data` endpoint for real-time ingestion
2. **Configure MQTT**: Forward sensor data to the API
3. **Monitor Database**: Check `/api/database-stats` for system health
4. **Analytics**: Query `live_sensor_data` table for insights
5. **Alerts**: Set up automated alerts on unresolved failures

---

## 📚 Documentation

- **`IMPLEMENTATION_SUMMARY.md`** - Recent SQLite integration & features
- **`SQLITE_DATABASE_GUIDE.md`** - Database API reference & usage
- **`TECHNICAL_DETAILS.md`** - Architecture & implementation details
- **`ARCHITECTURE.md`** - System design overview
- **`FEATURE_IMPLEMENTATION.md`** - Feature engineering details

---

## 📦 Dependencies

Key packages:

- **Flask** - Web framework
- **XGBoost, LightGBM** - ML models
- **scikit-learn** - ML utilities
- **pandas, numpy** - Data processing
- **SHAP** - Explainability
- **SQLAlchemy** - Database ORM

See `requirements.txt` for complete list.

---

## ⚠️ Important Notes

1. **Database Auto-Creation**: `models/factory_guard.db` is created automatically on first run
2. **No Migration Needed**: Works alongside existing CSV logs in parallel
3. **Backward Compatible**: All existing APIs continue to work
4. **Performance**: Indexed queries typically < 10ms
5. **Thread-Safe**: SQLite handles concurrent Flask requests

---

## 🔐 System Health Checklist

- [ ] Database initializes on app startup
- [ ] Can POST to `/api/live-sensor-data`
- [ ] New machines appear in "Machines at Risk"
- [ ] "Delete/Resolve" button persists data
- [ ] Page refresh retains resolved state
- [ ] `/api/database-stats` returns correct counts
- [ ] `models/factory_guard.db` file exists
- [ ] Dashboard loads all metrics

---

## 📝 License & Credits

**FactoryGuard AI** - Industrial Predictive Maintenance System

Built for real-time IoT sensor monitoring and machine failure prediction.

---

**Last Updated**: May 2026 | **Status**: ✅ Production Ready
