# SQLite Database Implementation - Update Guide

## Overview

This update replaces local storage (localStorage/CSV) with SQLite database for persistent, queryable storage of machine failures and their resolution status. This solves two critical issues:

1. **Cannot delete "Machines at Risk"** - Previously used localStorage which is volatile and frontend-dependent
2. **Live sensor data not captured** - machine_7845 failures weren't being logged because there was no endpoint for live sensor ingestion

## Architecture Changes

### Before
```
Live Sensors → Simulation → CSV Logs → Frontend localStorage → Display
(No real integration)                  (Volatile, lost on refresh)
```

### After
```
Live Sensors (HTTP) → /api/live-sensor-data → SQLite Database → Backend query → Display
                                                    ↓
                                            Persistent records
                                            Support real-time deletion
                                            Machine history tracking
```

## New Database Schema

### `machine_failures` Table
- **Purpose**: Log all failure predictions
- **Key Fields**:
  - `id`: Unique record ID
  - `machine_id`: Which machine failed
  - `prediction`: 0 (normal) or 1 (failure)
  - `failure_probability`: ML model confidence
  - `risk_level`: CRITICAL, HIGH, MEDIUM, LOW
  - `resolved`: Boolean flag for resolution status
  - `resolved_timestamp`: When it was marked resolved
  - `notes`: Maintenance notes
  - **Indexes**: On machine_id, timestamp, resolved, prediction

### `resolved_machines` Table
- **Purpose**: Track permanently resolved machines
- **Key Fields**:
  - `machine_id`: Machine that was resolved
  - `resolved_timestamp`: Resolution date/time
  - `resolved_by`: Who resolved it
  - `notes`: Resolution details
  - `reactivated`: If issue occurred again

### `live_sensor_data` Table
- **Purpose**: Archive all sensor readings
- **Key Fields**:
  - `machine_id`: Which machine
  - `air_temp`, `process_temp`, `rotational_speed`, `torque`, `tool_wear`
  - `source`: Where data came from (api, mqtt, simulation, etc.)
  - `metadata`: JSON field for extensibility

### `machine_registry` Table
- **Purpose**: Track all machines in the system
- **Key Fields**:
  - `machine_id`: Unique identifier
  - `created_timestamp`: First seen
  - `last_seen`: Most recent activity
  - `status`: active/inactive
  - `metadata`: JSON for location, type, etc.

## New API Endpoints

### 1. POST `/api/live-sensor-data` - Ingest Real-Time Sensor Data

**Purpose**: Accept live sensor readings from IoT devices, MQTT brokers, or hardware gateways

**Request Body**:
```json
{
  "machine_id": "machine_7845",
  "air_temp": 42.5,
  "process_temp": 65.3,
  "rotational_speed": 3200,
  "torque": 68.5,
  "tool_wear": 215,
  "source": "iot_gateway",
  "metadata": {
    "location": "Assembly Line 5",
    "gateway_id": "gateway_01"
  }
}
```

**Response**:
```json
{
  "success": true,
  "machine_id": "machine_7845",
  "prediction": 1,
  "failure_probability": 0.8234,
  "risk_level": "CRITICAL",
  "maintenance_required": true,
  "timestamp": "2026-05-13T14:30:45.123456"
}
```

**Key Features**:
- Automatically registers new machines
- Logs sensor reading to database
- Runs ML prediction
- Returns risk assessment
- Data persists for analysis

### 2. POST `/api/resolve-machine` - Mark Machine as Resolved

**Purpose**: Permanently resolve a machine failure (won't show in "Machines at Risk" anymore)

**Request Body**:
```json
{
  "machine_id": "machine_7845",
  "resolved_by": "maintenance_team",
  "notes": "Replaced worn tool, completed calibration"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Machine machine_7845 resolved",
  "machine_id": "machine_7845"
}
```

**Key Features**:
- ✅ **Persistent** - Uses SQLite, survives page refresh
- ✅ **Auditable** - Records who resolved and when
- ✅ **Revertible** - Can reactivate if issue recurs
- ✅ **Real-time** - Removed from "Machines at Risk" immediately

### 3. GET `/api/machines-at-risk` - Updated to Use SQLite

**Now Returns**:
- Only unresolved machine failures
- Only queries resolved machines from database (not localStorage)
- Grouped by latest reading per machine
- Sorted by risk score

### 4. GET `/api/database-stats` - Monitor Database

**Response**:
```json
{
  "success": true,
  "stats": {
    "total_failures": 1247,
    "unresolved_failures": 23,
    "total_machines": 50,
    "risk_distribution": {
      "CRITICAL": 5,
      "HIGH": 8,
      "MEDIUM": 7,
      "LOW": 3
    }
  }
}
```

## How It Solves Your Issues

### Issue #1: Can't Delete "Machines at Risk"

**Before**: 
- Deletion was stored only in browser localStorage
- Lost on page refresh/new browser/different device
- No backend visibility

**After**:
```
User clicks "Delete/Resolve" 
  ↓
Frontend calls `/api/resolve-machine` API
  ↓
Backend updates `machine_failures.resolved = 1`
  ↓
Backend updates `resolved_machines` table
  ↓
Frontend refreshes and queries backend
  ↓
Backend only returns unresolved failures
  ↓
Machine stays deleted even after refresh! ✅
```

### Issue #2: machine_7845 Not Logged

**Before**:
- Only simulated data from check.csv was being processed
- Real sensor data had nowhere to go
- machine_7845 failure would be lost

**After**:
```
IoT Device reads machine_7845 sensors
  ↓
POST to `/api/live-sensor-data` with readings
  ↓
Backend logs sensor data to database
  ↓
Backend runs ML prediction
  ↓
If prediction = 1 (failure), logged to machine_failures table
  ↓
Immediately appears in "Machines at Risk" ✅
  ↓
Data persists for historical analysis ✅
```

## Usage Examples

### Python Client Example
```python
import requests

# Send real-time sensor data
response = requests.post(
    'http://localhost:5000/api/live-sensor-data',
    json={
        'machine_id': 'machine_7845',
        'air_temp': 42.5,
        'process_temp': 65.3,
        'rotational_speed': 3200,
        'torque': 68.5,
        'tool_wear': 215,
        'source': 'iot_gateway'
    }
)

result = response.json()
print(f"Machine: {result['machine_id']}")
print(f"Risk: {result['risk_level']}")
```

Run `python live_sensor_example.py` for complete examples.

### cURL Example
```bash
# Send sensor data
curl -X POST http://localhost:5000/api/live-sensor-data \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "machine_7845",
    "air_temp": 42.5,
    "process_temp": 65.3,
    "rotational_speed": 3200,
    "torque": 68.5,
    "tool_wear": 215
  }'

# Resolve machine
curl -X POST http://localhost:5000/api/resolve-machine \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "machine_7845",
    "resolved_by": "maintenance",
    "notes": "Replaced worn components"
  }'
```

### JavaScript/Fetch Example
```javascript
// Send sensor data
fetch('/api/live-sensor-data', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    machine_id: 'machine_7845',
    air_temp: 42.5,
    process_temp: 65.3,
    rotational_speed: 3200,
    torque: 68.5,
    tool_wear: 215
  })
})
.then(r => r.json())
.then(data => console.log(data.risk_level));
```

## Integration Points

### 1. IoT/MQTT Gateway Integration
```python
import paho.mqtt.client as mqtt
import requests

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload)
    
    # Forward to FactoryGuard API
    requests.post('http://localhost:5000/api/live-sensor-data', 
                  json=payload)

client = mqtt.Client()
client.on_message = on_message
client.connect("mqtt.example.com", 1883)
client.subscribe("factory/machines/+/sensors")
client.loop_forever()
```

### 2. Time-Series Database (Future Enhancement)
The `live_sensor_data` table now stores all readings - can be queried for trends:
```python
db.get_latest_sensor_reading("machine_7845")  # Most recent
df.query('machine_id == "machine_7845"')       # All readings (historical analysis)
```

### 3. Dashboard Refresh
Frontend automatically uses database-backed resolution:
- No localStorage needed
- Works across devices/browsers
- Real-time sync

## File Changes

### New Files
- **`src/database.py`** - SQLite manager with 250+ lines of database logic

### Modified Files
- **`app.py`** - Added database initialization and new API endpoints
- **`static/js/dashboard.js`** - Updated resolve mechanism to use API

### No Changes Needed
- `src/prediction_logger.py` - Still works for CSV logging (can be deprecated)
- `src/factory_health.py` - Unchanged
- Other modules - Unchanged

## Database File Location

Default: `models/factory_guard.db`

To change, update in `app.py`:
```python
db = init_db("custom/path/factory_guard.db")
```

## Data Persistence

- ✅ Survives server restart
- ✅ Survives browser refresh
- ✅ Survives page navigation
- ✅ Survives client-side cache clear
- ✅ Queryable for analytics
- ✅ Auditable (timestamps, who resolved)

## Performance Considerations

- Indexes created on frequently-queried columns
- `machine_failures` queries limited to `LIMIT 500` by default
- Database operations are synchronous (< 10ms typical)
- Can be upgraded to async with connection pooling if needed

## Backward Compatibility

- ✅ Existing CSV logs still created (`prediction_history.csv`)
- ✅ Old localStorage-based approach removed (deprecated)
- ✅ All existing API endpoints still work
- ✅ New endpoints are additions, not replacements

## Troubleshooting

### Database locked error
```
SQLite locks if multiple writes happen simultaneously.
Flask handles this with locks. If persists, ensure single app instance.
```

### Machine not appearing in "Machines at Risk"
1. Ensure prediction = 1 (failure)
2. Ensure resolved = 0 (not marked resolved)
3. Check: `GET /api/database-stats`
4. Check database directly: `sqlite3 models/factory_guard.db`

### How to manually check database
```bash
# Terminal
sqlite3 models/factory_guard.db

# In SQLite shell
SELECT * FROM machine_failures WHERE prediction = 1 AND resolved = 0;
SELECT * FROM resolved_machines;
.quit
```

## Next Steps

1. **Test live sensor ingestion** - Run `python live_sensor_example.py`
2. **Connect IoT devices** - Point to `POST /api/live-sensor-data`
3. **Monitor database** - Check `GET /api/database-stats`
4. **Verify deletions persist** - Resolve machine, refresh page
5. **Analyze trends** - Query `live_sensor_data` for machine history

## Questions?

Refer to:
- `src/database.py` - Full database implementation
- `live_sensor_example.py` - Usage examples
- API responses include error messages for debugging
