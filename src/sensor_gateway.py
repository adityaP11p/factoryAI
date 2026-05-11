"""
FactoryGuard AI - Sensor Gateway Integration
Handles data ingestion from multiple sources: CSV uploads, MQTT brokers, HTTP APIs, and hardware pins.
"""

import os
import csv
import json
import time
import threading
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SensorGateway:
    """Unified sensor data gateway supporting multiple ingestion modes."""
    
    def __init__(self, data_dir="data/raw"):
        """Initialize gateway with data directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Live data from different sources
        self.live_data = {}  # {machine_id: latest_reading}
        self.data_buffer = []  # Buffer for batch processing
        
        # Gateway status
        self.is_running = False
        self.gateway_mode = None
        
    # ─── CSV UPLOAD MODE ───────────────────────────────────────────────
    
    def upload_csv(self, filepath, machine_id_prefix="MACHINE"):
        """
        Upload CSV file containing sensor data.
        Expects columns: air_temp, process_temp, rotational_speed, torque, tool_wear, [optional: Type]
        """
        try:
            logger.info(f"Uploading CSV from: {filepath}")
            
            df = pd.read_csv(filepath)
            
            # Validate required columns
            required_cols = ["air_temp", "process_temp", "rotational_speed", "torque", "tool_wear"]
            missing = set(required_cols) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            # Add machine_id if not present
            if 'machine_id' not in df.columns:
                df['machine_id'] = f"{machine_id_prefix}_{1:03d}"
                logger.info(f"Added default machine_id: {df['machine_id'].iloc[0]}")
            
            # Add timestamp if not present
            if 'timestamp' not in df.columns:
                df['timestamp'] = datetime.now().isoformat()
            
            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()
            
            # Save to central storage
            output_path = self.data_dir / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(output_path, index=False)
            
            logger.info(f"✅ Uploaded {len(df)} records to {output_path}")
            return {
                "success": True,
                "records": len(df),
                "machines": df['machine_id'].unique().tolist() if 'machine_id' in df.columns else [],
                "file": str(output_path)
            }
        except Exception as e:
            logger.error(f"CSV upload error: {e}")
            return {"success": False, "error": str(e)}
    
    # ─── REAL-TIME GATEWAY MODE ────────────────────────────────────────
    
    def start_mqtt_gateway(self, broker_address="localhost", port=1883, topic_prefix="factory/machines"):
        """
        Start MQTT subscriber for real-time sensor data.
        
        Expected MQTT topic structure:
        factory/machines/{machine_id}/sensors -> JSON: {air_temp, process_temp, ...}
        """
        try:
            import paho.mqtt.client as mqtt
            
            logger.info(f"Initializing MQTT gateway: {broker_address}:{port}")
            
            client = mqtt.Client()
            
            def on_connect(client, userdata, flags, rc):
                if rc == 0:
                    logger.info("✅ MQTT connected successfully")
                    client.subscribe(f"{topic_prefix}/#")
                else:
                    logger.error(f"MQTT connection failed with code {rc}")
            
            def on_message(client, userdata, msg):
                """Process incoming MQTT message."""
                try:
                    # Parse topic: factory/machines/{machine_id}/sensors
                    parts = msg.topic.split('/')
                    if len(parts) >= 3:
                        machine_id = parts[2]
                        
                        # Parse sensor data
                        sensor_data = json.loads(msg.payload.decode())
                        sensor_data['machine_id'] = machine_id
                        sensor_data['timestamp'] = datetime.now().isoformat()
                        
                        # Store in live data
                        self.live_data[machine_id] = sensor_data
                        self.data_buffer.append(sensor_data)
                        
                        logger.info(f"📊 Received from {machine_id}: {sensor_data}")
                except Exception as e:
                    logger.error(f"MQTT message error: {e}")
            
            client.on_connect = on_connect
            client.on_message = on_message
            
            # Connect in background thread
            def connect_mqtt():
                client.connect(broker_address, port, keepalive=60)
                client.loop_forever()
            
            mqtt_thread = threading.Thread(target=connect_mqtt, daemon=True)
            mqtt_thread.start()
            
            self.is_running = True
            self.gateway_mode = "MQTT"
            logger.info("✅ MQTT gateway started")
            
            return {"success": True, "mode": "MQTT", "broker": broker_address}
        
        except ImportError:
            logger.error("paho-mqtt not installed. Install with: pip install paho-mqtt")
            return {"success": False, "error": "paho-mqtt not installed"}
        except Exception as e:
            logger.error(f"MQTT gateway error: {e}")
            return {"success": False, "error": str(e)}
    
    def start_api_gateway(self, api_endpoint, poll_interval=60, machine_ids=None):
        """
        Start API poller for sensor data from cloud platform.
        
        Expected API response:
        {
            "machines": [
                {"machine_id": "M1", "air_temp": 298.5, "process_temp": 310.2, ...},
                ...
            ]
        }
        """
        try:
            import requests
            
            logger.info(f"Initializing API gateway: {api_endpoint} (poll every {poll_interval}s)")
            
            if machine_ids is None:
                machine_ids = ["M1", "M2", "M3"]  # Default machines
            
            def poll_api():
                while self.is_running:
                    try:
                        # Fetch from API
                        response = requests.get(api_endpoint, params={"machines": ",".join(machine_ids)}, timeout=10)
                        response.raise_for_status()
                        
                        data = response.json()
                        
                        # Process each machine's data
                        for machine_data in data.get("machines", []):
                            machine_id = machine_data.get("machine_id")
                            machine_data['timestamp'] = datetime.now().isoformat()
                            
                            self.live_data[machine_id] = machine_data
                            self.data_buffer.append(machine_data)
                            
                            logger.info(f"📡 API data for {machine_id}")
                        
                        time.sleep(poll_interval)
                    
                    except Exception as e:
                        logger.warning(f"API poll error: {e}, retrying in {poll_interval}s")
                        time.sleep(poll_interval)
            
            # Start polling in background
            api_thread = threading.Thread(target=poll_api, daemon=True)
            api_thread.start()
            
            self.is_running = True
            self.gateway_mode = "API"
            logger.info("✅ API gateway started")
            
            return {"success": True, "mode": "API", "endpoint": api_endpoint, "machines": machine_ids}
        
        except ImportError:
            logger.error("requests not installed")
            return {"success": False, "error": "requests not installed"}
        except Exception as e:
            logger.error(f"API gateway error: {e}")
            return {"success": False, "error": str(e)}
    
    def start_hardware_gateway(self, gpio_pins=None, poll_interval=2):
        """
        Start hardware GPIO poller for Raspberry Pi / Arduino sensor reads.
        
        gpio_pins: {
            "M1": {"temp": GPIO_PIN_1, "rpm": GPIO_PIN_2, ...},
            "M2": {...}
        }
        """
        try:
            import RPi.GPIO as GPIO
            
            logger.info("Initializing Hardware GPIO gateway (Raspberry Pi)")
            
            if gpio_pins is None:
                gpio_pins = {}
            
            GPIO.setmode(GPIO.BCM)
            
            def read_hardware():
                while self.is_running:
                    try:
                        for machine_id, pins in gpio_pins.items():
                            # Simulated hardware read (replace with actual GPIO code)
                            sensor_data = {
                                "machine_id": machine_id,
                                "air_temp": 298.0 + (GPIO.input(pins.get("temp", 17)) or 0) * 10,
                                "process_temp": 310.0 + (GPIO.input(pins.get("process", 27)) or 0) * 10,
                                "rotational_speed": 1500 + (GPIO.input(pins.get("rpm", 22)) or 0) * 100,
                                "torque": 40.0 + (GPIO.input(pins.get("torque", 23)) or 0) * 5,
                                "tool_wear": 100 + (GPIO.input(pins.get("wear", 24)) or 0) * 20,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            self.live_data[machine_id] = sensor_data
                            self.data_buffer.append(sensor_data)
                        
                        time.sleep(poll_interval)
                    except Exception as e:
                        logger.warning(f"Hardware read error: {e}")
                        time.sleep(poll_interval)
            
            hw_thread = threading.Thread(target=read_hardware, daemon=True)
            hw_thread.start()
            
            self.is_running = True
            self.gateway_mode = "HARDWARE"
            logger.info("✅ Hardware gateway started")
            
            return {"success": True, "mode": "HARDWARE", "machines": list(gpio_pins.keys())}
        
        except ImportError:
            logger.error("RPi.GPIO not available (not on Raspberry Pi)")
            return {"success": False, "error": "RPi.GPIO not available"}
        except Exception as e:
            logger.error(f"Hardware gateway error: {e}")
            return {"success": False, "error": str(e)}
    
    # ─── COMMON METHODS ───────────────────────────────────────────────
    
    def get_latest_data(self, machine_id=None):
        """Get latest sensor reading for machine or all machines."""
        if machine_id:
            return self.live_data.get(machine_id, {})
        return self.live_data
    
    def get_buffered_data(self):
        """Get and clear buffered data."""
        data = self.data_buffer.copy()
        self.data_buffer.clear()
        return data
    
    def save_buffer_to_csv(self):
        """Save buffer to CSV file."""
        if not self.data_buffer:
            logger.warning("No data in buffer to save")
            return None
        
        df = pd.DataFrame(self.data_buffer)
        output_path = self.data_dir / f"gateway_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"💾 Saved {len(df)} records to {output_path}")
        self.data_buffer.clear()
        return str(output_path)
    
    def stop(self):
        """Stop all gateway operations."""
        self.is_running = False
        logger.info(f"⏹️  Gateway stopped ({self.gateway_mode} mode)")


# Global gateway instance
_gateway = None

def get_gateway():
    """Get or create global gateway instance."""
    global _gateway
    if _gateway is None:
        _gateway = SensorGateway()
    return _gateway
