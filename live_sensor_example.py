"""
Live Sensor Data Ingestion Example
Demonstrates how to send real-time sensor data from hardware/IoT devices to FactoryGuard
"""

import requests
import json
from datetime import datetime

# API Base URL (update this if your server is on a different host/port)
API_BASE_URL = "http://localhost:5000"

# Example 1: Ingest sensor data for machine_7845
def ingest_sensor_data_machine_7845():
    """Send live sensor data for machine_7845 with failure condition."""
    
    sensor_payload = {
        "machine_id": "machine_7845",
        "air_temp": 42.5,          # Higher than normal
        "process_temp": 65.3,      # Higher than normal
        "rotational_speed": 3200,   # High speed
        "torque": 68.5,            # High torque
        "tool_wear": 215,          # High wear
        "source": "iot_gateway",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "location": "Assembly Line 5",
            "gateway_id": "gateway_01"
        }
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/live-sensor-data",
            json=sensor_payload,
            headers={"Content-Type": "application/json"}
        )
        
        result = response.json()
        if result['success']:
            print(f"✅ Successfully ingested data for {sensor_payload['machine_id']}")
            print(f"   Prediction: {'FAILURE' if result['prediction'] == 1 else 'NORMAL'}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Failure Probability: {result['failure_probability']:.4f}")
            return True
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


# Example 2: Ingest sensor data for multiple machines
def ingest_batch_sensor_data(machines_data):
    """Send sensor data for multiple machines."""
    
    results = []
    for machine_data in machines_data:
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/live-sensor-data",
                json=machine_data,
                headers={"Content-Type": "application/json"}
            )
            
            result = response.json()
            results.append({
                "machine_id": machine_data['machine_id'],
                "success": result['success'],
                "prediction": result.get('prediction'),
                "risk_level": result.get('risk_level')
            })
        except Exception as e:
            results.append({
                "machine_id": machine_data['machine_id'],
                "success": False,
                "error": str(e)
            })
    
    return results


# Example 3: Resolve a machine failure
def resolve_machine_failure(machine_id):
    """Mark a machine failure as resolved."""
    
    payload = {
        "machine_id": machine_id,
        "resolved_by": "maintenance_team",
        "notes": "Completed routine maintenance"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/resolve-machine",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        result = response.json()
        if result['success']:
            print(f"✅ Machine {machine_id} marked as resolved")
            return True
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


# Example 4: Get database statistics
def get_database_statistics():
    """Fetch database statistics."""
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/database-stats",
            headers={"Content-Type": "application/json"}
        )
        
        result = response.json()
        if result['success']:
            stats = result['stats']
            print("📊 Database Statistics:")
            print(f"   Total Failures: {stats.get('total_failures', 0)}")
            print(f"   Unresolved Failures: {stats.get('unresolved_failures', 0)}")
            print(f"   Total Machines: {stats.get('total_machines', 0)}")
            print(f"   Risk Distribution: {stats.get('risk_distribution', {})}")
            return stats
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("FactoryGuard AI - Live Sensor Data Ingestion Examples")
    print("=" * 60 + "\n")
    
    # Example 1: Send data for machine_7845
    print("1️⃣  Ingesting sensor data for machine_7845...")
    ingest_sensor_data_machine_7845()
    
    print("\n" + "-" * 60 + "\n")
    
    # Example 2: Send batch data
    print("2️⃣  Ingesting batch sensor data for multiple machines...")
    batch_data = [
        {
            "machine_id": "machine_1001",
            "air_temp": 35.2,
            "process_temp": 52.1,
            "rotational_speed": 2800,
            "torque": 42.3,
            "tool_wear": 95,
            "source": "mqtt_broker"
        },
        {
            "machine_id": "machine_1002",
            "air_temp": 38.5,
            "process_temp": 58.9,
            "rotational_speed": 3100,
            "torque": 55.8,
            "tool_wear": 180,
            "source": "mqtt_broker"
        }
    ]
    
    results = ingest_batch_sensor_data(batch_data)
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['machine_id']}: {result.get('risk_level', 'ERROR')}")
    
    print("\n" + "-" * 60 + "\n")
    
    # Example 3: Get statistics
    print("3️⃣  Fetching database statistics...")
    get_database_statistics()
    
    print("\n" + "-" * 60 + "\n")
    
    # Example 4: Resolve a machine (if any failures exist)
    print("4️⃣  Example: Resolving a machine failure...")
    print("   (Uncomment to test, replace machine_id with actual failure)")
    # resolve_machine_failure("machine_1001")
    
    print("\n" + "=" * 60)
    print("✅ Examples completed!")
    print("=" * 60)
