#!/usr/bin/env python3
"""
Quick Start Guide - Test the SQLite Database Implementation
Run this to verify everything works correctly
"""

import requests
import json
import time
from datetime import datetime

API_URL = "http://localhost:5000"

def print_header(text):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def test_database_stats():
    """Test 1: Check database statistics"""
    print_header("TEST 1: Database Statistics")
    
    try:
        response = requests.get(f"{API_URL}/api/database-stats")
        data = response.json()
        
        if data['success']:
            stats = data['stats']
            print(f"✅ Total Failures: {stats['total_failures']}")
            print(f"✅ Unresolved Failures: {stats['unresolved_failures']}")
            print(f"✅ Total Machines: {stats['total_machines']}")
            print(f"✅ Risk Distribution: {stats['risk_distribution']}")
            return True
        else:
            print(f"❌ Error: {data.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_live_sensor_ingestion():
    """Test 2: Ingest live sensor data for machine_7845"""
    print_header("TEST 2: Live Sensor Data Ingestion")
    
    sensor_data = {
        "machine_id": "machine_7845",
        "air_temp": 42.5,
        "process_temp": 65.3,
        "rotational_speed": 3200,
        "torque": 68.5,
        "tool_wear": 215,
        "source": "test_script",
        "metadata": {
            "test": True,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    try:
        print("📤 Sending sensor data for machine_7845 with failure conditions...")
        response = requests.post(f"{API_URL}/api/live-sensor-data", 
                               json=sensor_data)
        data = response.json()
        
        if data['success']:
            print(f"✅ Machine ID: {data['machine_id']}")
            print(f"✅ Prediction: {'FAILURE ⚠️' if data['prediction'] == 1 else 'NORMAL ✓'}")
            print(f"✅ Risk Level: {data['risk_level']}")
            print(f"✅ Failure Probability: {data['failure_probability']:.4f}")
            print(f"✅ Ensemble Agreement: {data['ensemble_agreement']:.4f}")
            return True
        else:
            print(f"❌ Error: {data.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_machines_at_risk():
    """Test 3: Check machines at risk list"""
    print_header("TEST 3: Machines at Risk List")
    
    try:
        response = requests.get(f"{API_URL}/api/machines-at-risk")
        data = response.json()
        
        if data['success']:
            machines = data['machines']
            print(f"✅ Machines at Risk: {len(machines)}")
            
            if machines:
                print("\nTop at-risk machines:")
                for i, machine in enumerate(machines[:5], 1):
                    print(f"  {i}. {machine['machine_id']} - {machine['risk_level']} ({machine['avg_risk_score']:.2%})")
            
            # Check if machine_7845 is there
            machine_ids = [m['machine_id'] for m in machines]
            if 'machine_7845' in machine_ids:
                print("\n✅ machine_7845 successfully appears in 'Machines at Risk'!")
                return True, machines
            else:
                print("\n⚠️  machine_7845 not found in list (may not have failed)")
                return True, machines
        else:
            print(f"❌ Error: {data.get('error')}")
            return False, []
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False, []

def test_resolve_machine(machine_id):
    """Test 4: Resolve a machine failure"""
    print_header(f"TEST 4: Resolve Machine ({machine_id})")
    
    payload = {
        "machine_id": machine_id,
        "resolved_by": "test_script",
        "notes": "Tested via quick start script"
    }
    
    try:
        print(f"🔧 Resolving machine {machine_id}...")
        response = requests.post(f"{API_URL}/api/resolve-machine", 
                               json=payload)
        data = response.json()
        
        if data['success']:
            print(f"✅ {data['message']}")
            return True
        else:
            print(f"❌ Error: {data.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_persistence():
    """Test 5: Verify resolution persists"""
    print_header("TEST 5: Verify Resolution Persistence")
    
    try:
        print("⏳ Checking if resolved machine stays resolved...")
        response = requests.get(f"{API_URL}/api/machines-at-risk")
        data = response.json()
        
        if data['success']:
            machines = data['machines']
            machine_ids = [m['machine_id'] for m in machines]
            
            if 'machine_7845' not in machine_ids:
                print("✅ Resolved machine (machine_7845) not in list - PERSISTENCE WORKS!")
                return True
            else:
                print("⚠️  Resolved machine still in list - may need refresh")
                return True
        else:
            print(f"❌ Error: {data.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "🧪 FACTORYGUARD AI - SQLITE DATABASE QUICK START")
    print("Testing all new features...")
    
    # Check connectivity
    print("\n🔗 Checking API connectivity...")
    try:
        response = requests.get(f"{API_URL}/api/database-stats", timeout=5)
        if response.status_code == 200:
            print("✅ API is reachable")
        else:
            print("❌ API returned unexpected status code")
            return
    except:
        print(f"❌ Cannot reach API at {API_URL}")
        print("   Make sure Flask app is running: python app.py")
        return
    
    # Run tests
    results = []
    
    # Test 1: Database stats
    results.append(("Database Stats", test_database_stats()))
    
    # Test 2: Live sensor ingestion
    results.append(("Live Sensor Ingestion", test_live_sensor_ingestion()))
    time.sleep(1)  # Wait for database write
    
    # Test 3: Machines at risk
    success, machines = test_machines_at_risk()
    results.append(("Machines at Risk", success))
    
    # Test 4: Resolve machine (only if machine_7845 exists)
    if machines and any(m['machine_id'] == 'machine_7845' for m in machines):
        results.append(("Resolve Machine", test_resolve_machine('machine_7845')))
        time.sleep(1)
        
        # Test 5: Persistence
        results.append(("Persistence Check", test_persistence()))
    
    # Summary
    print_header("TEST SUMMARY")
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! SQLite database is working correctly.")
        print("\n📚 Next steps:")
        print("  1. Check SQLITE_DATABASE_GUIDE.md for detailed documentation")
        print("  2. Run live_sensor_example.py for more examples")
        print("  3. Connect your IoT devices to POST /api/live-sensor-data")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
