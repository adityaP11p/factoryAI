"""
SQLite Database Module for FactoryGuard AI
Handles persistent storage of:
- Machine failure predictions
- Resolved status for machines at risk
- Live sensor data ingestion
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FactoryDatabase:
    """SQLite database manager for FactoryGuard predictions and alerts."""
    
    def __init__(self, db_path: str = "models/factory_guard.db"):
        """Initialize database path."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def get_connection(self):
        """Create a new database connection for each operation."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Initialize database schema."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Table: Machine Failure Predictions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS machine_failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    machine_id TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    failure_probability REAL,
                    normal_probability REAL,
                    risk_level TEXT,
                    air_temp REAL,
                    process_temp REAL,
                    rotational_speed REAL,
                    torque REAL,
                    tool_wear REAL,
                    model_name TEXT,
                    ensemble_agreement REAL,
                    resolved BOOLEAN DEFAULT 0,
                    resolved_timestamp DATETIME,
                    resolved_by TEXT,
                    notes TEXT,
                    UNIQUE(timestamp, machine_id) ON CONFLICT REPLACE
                )
            ''')
            
            # Table: Resolved Machine Status
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS resolved_machines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    machine_id TEXT NOT NULL UNIQUE,
                    resolved_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resolved_by TEXT DEFAULT 'system',
                    notes TEXT,
                    reactivated BOOLEAN DEFAULT 0,
                    reactivated_timestamp DATETIME
                )
            ''')
            
            # Table: Live Sensor Data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_sensor_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    machine_id TEXT NOT NULL,
                    air_temp REAL,
                    process_temp REAL,
                    rotational_speed REAL,
                    torque REAL,
                    tool_wear REAL,
                    source TEXT DEFAULT 'api',
                    metadata JSON
                )
            ''')
            
            # Table: Machine Registry (for tracking all machines)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS machine_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    machine_id TEXT NOT NULL UNIQUE,
                    created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_seen DATETIME,
                    status TEXT DEFAULT 'active',
                    location TEXT,
                    metadata JSON
                )
            ''')
            
            # Create indexes for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_machine_id ON machine_failures(machine_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON machine_failures(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_resolved ON machine_failures(resolved)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_prediction ON machine_failures(prediction)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sensor_machine ON live_sensor_data(machine_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sensor_timestamp ON live_sensor_data(timestamp)')
            
            conn.commit()
            logger.info(f"✅ Database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def log_failure_prediction(self, 
                              machine_id: str,
                              prediction: int,
                              failure_probability: float,
                              normal_probability: float,
                              risk_level: str,
                              sensor_data: Dict[str, Any],
                              model_info: Dict[str, Any]) -> int:
        """Log a failure prediction to database."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Register machine if new
            self.register_machine(machine_id)
            
            cursor.execute('''
                INSERT INTO machine_failures (
                    machine_id, prediction, failure_probability, normal_probability,
                    risk_level, air_temp, process_temp, rotational_speed, torque, tool_wear,
                    model_name, ensemble_agreement, resolved
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                machine_id,
                prediction,
                round(failure_probability, 4),
                round(normal_probability, 4),
                risk_level,
                sensor_data.get('air_temp', 0),
                sensor_data.get('process_temp', 0),
                sensor_data.get('rotational_speed', 0),
                sensor_data.get('torque', 0),
                sensor_data.get('tool_wear', 0),
                model_info.get('model_name', 'Unknown'),
                model_info.get('ensemble_agreement', 0),
                0  # resolved = False
            ))
            
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error logging failure prediction: {e}")
            conn.rollback()
            return -1
        finally:
            conn.close()
    
    def get_unresolved_failures(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Get all unresolved machine failures."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    id, timestamp, machine_id, prediction, failure_probability,
                    normal_probability, risk_level, air_temp, process_temp, 
                    rotational_speed, torque, tool_wear, model_name, 
                    ensemble_agreement, resolved
                FROM machine_failures
                WHERE resolved = 0 AND prediction = 1
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching unresolved failures: {e}")
            return []
        finally:
            conn.close()
    
    def resolve_machine_failure(self, machine_id: str, notes: str = "", resolved_by: str = "system") -> bool:
        """Mark a machine failure as resolved."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Update machine_failures table
            cursor.execute('''
                UPDATE machine_failures
                SET resolved = 1, resolved_timestamp = CURRENT_TIMESTAMP, 
                    resolved_by = ?, notes = ?
                WHERE machine_id = ? AND resolved = 0 AND prediction = 1
            ''', (resolved_by, notes, machine_id))
            
            # Insert/update resolved_machines table
            cursor.execute('''
                INSERT OR REPLACE INTO resolved_machines (machine_id, resolved_by, notes)
                VALUES (?, ?, ?)
            ''', (machine_id, resolved_by, notes))
            
            conn.commit()
            logger.info(f"✅ Machine {machine_id} marked as resolved")
            return True
        except Exception as e:
            logger.error(f"Error resolving machine: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def is_machine_resolved(self, machine_id: str) -> bool:
        """Check if a machine failure is resolved."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id FROM resolved_machines
                WHERE machine_id = ? AND reactivated = 0
            ''', (machine_id,))
            
            return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking machine resolution status: {e}")
            return False
        finally:
            conn.close()
    
    def register_machine(self, machine_id: str, location: str = "", metadata: Dict = None) -> bool:
        """Register a new machine in the registry."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            metadata_json = json.dumps(metadata) if metadata else None
            cursor.execute('''
                INSERT OR IGNORE INTO machine_registry (machine_id, location, metadata)
                VALUES (?, ?, ?)
            ''', (machine_id, location, metadata_json))
            
            # Update last_seen
            cursor.execute('''
                UPDATE machine_registry
                SET last_seen = CURRENT_TIMESTAMP
                WHERE machine_id = ?
            ''', (machine_id,))
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error registering machine: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def log_live_sensor_reading(self, machine_id: str, sensor_data: Dict[str, float],
                               source: str = "api", metadata: Dict = None) -> int:
        """Log a live sensor reading."""
        conn = self.get_connection()
        try:
            # Register machine if new
            self.register_machine(machine_id)
            
            metadata_json = json.dumps(metadata) if metadata else None
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO live_sensor_data (
                    machine_id, air_temp, process_temp, rotational_speed, 
                    torque, tool_wear, source, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                machine_id,
                sensor_data.get('air_temp', 0),
                sensor_data.get('process_temp', 0),
                sensor_data.get('rotational_speed', 0),
                sensor_data.get('torque', 0),
                sensor_data.get('tool_wear', 0),
                source,
                metadata_json
            ))
            
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error logging sensor reading: {e}")
            conn.rollback()
            return -1
        finally:
            conn.close()
    
    def get_latest_sensor_reading(self, machine_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest sensor reading for a machine."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM live_sensor_data
                WHERE machine_id = ? 
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (machine_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error fetching latest sensor reading: {e}")
            return None
        finally:
            conn.close()
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Total failures
            cursor.execute('SELECT COUNT(*) FROM machine_failures WHERE prediction = 1')
            total_failures = cursor.fetchone()[0]
            
            # Unresolved failures
            cursor.execute('SELECT COUNT(*) FROM machine_failures WHERE resolved = 0 AND prediction = 1')
            unresolved_failures = cursor.fetchone()[0]
            
            # Risk distribution
            cursor.execute('''
                SELECT risk_level, COUNT(*) as count
                FROM machine_failures
                WHERE resolved = 0 AND prediction = 1
                GROUP BY risk_level
            ''')
            risk_dist = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Machines registered
            cursor.execute('SELECT COUNT(*) FROM machine_registry')
            total_machines = cursor.fetchone()[0]
            
            return {
                "total_failures": total_failures,
                "unresolved_failures": unresolved_failures,
                "risk_distribution": risk_dist,
                "total_machines": total_machines
            }
        except Exception as e:
            logger.error(f"Error fetching statistics: {e}")
            return {}
        finally:
            conn.close()
    
    def clear_resolved_history(self, days: int = 7) -> int:
        """Clear resolved failures older than N days."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM machine_failures
                WHERE resolved = 1 
                AND resolved_timestamp < datetime('now', '-' || ? || ' days')
            ''', (days,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(f"Cleared {deleted_count} resolved failure records")
            return deleted_count
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()
    
    def close(self):
        """Close database connection. (No-op since connections are per-operation)"""
        pass


# Global database instance
_db_instance = None


def get_database(db_path: str = "models/factory_guard.db") -> FactoryDatabase:
    """Get or create global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = FactoryDatabase(db_path)
    return _db_instance


def init_db(db_path: str = "models/factory_guard.db") -> FactoryDatabase:
    """Initialize database."""
    global _db_instance
    _db_instance = FactoryDatabase(db_path)
    return _db_instance
