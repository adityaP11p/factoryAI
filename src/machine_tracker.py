"""
Machine Tracker - Track individual machine data and predictions
Maintains a registry of machines and their historical performance metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MachineTracker:
    """Track individual machine metrics and performance history."""
    
    def __init__(self, data_dir: str = "models", n_machines: int = 50):
        """
        Initialize machine tracker.
        
        Args:
            data_dir: Directory to store machine data
            n_machines: Total number of machines to track
        """
        self.data_dir = Path(data_dir)
        self.n_machines = n_machines
        self.machines_file = self.data_dir / "machines_registry.json"
        self.predictions_dir = self.data_dir / "machine_predictions"
        self.predictions_dir.mkdir(exist_ok=True)
        
        self._initialize_machines()
    
    def _initialize_machines(self):
        """Initialize machine registry if not exists."""
        if not self.machines_file.exists():
            machines = {}
            for i in range(1, self.n_machines + 1):
                machines[f"machine_{i}"] = {
                    "machine_id": i,
                    "name": f"Machine-{i}",
                    "location": f"Floor {(i-1)//10 + 1}, Section {(i-1)%10 + 1}",
                    "type": "Industrial Robot Arm" if i % 3 == 0 else ("CNC Machine" if i % 3 == 1 else "Hydraulic Press"),
                    "status": "OPERATIONAL",
                    "installed_date": (datetime.now() - timedelta(days=np.random.randint(100, 1000))).isoformat(),
                    "last_maintenance": (datetime.now() - timedelta(days=np.random.randint(10, 100))).isoformat(),
                    "next_scheduled_maintenance": None,
                    "total_operating_hours": np.random.randint(10000, 100000),
                    "total_failures": np.random.randint(0, 50),
                    "creation_timestamp": datetime.now().isoformat()
                }
            self._save_machines(machines)
    
    def _load_machines(self) -> Dict[str, Any]:
        """Load machines registry."""
        try:
            if self.machines_file.exists():
                with open(self.machines_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading machines registry: {e}")
        return {}
    
    def _save_machines(self, machines: Dict[str, Any]):
        """Save machines registry."""
        try:
            with open(self.machines_file, 'w') as f:
                json.dump(machines, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving machines registry: {e}")
    
    def get_machine(self, machine_id: int) -> Optional[Dict[str, Any]]:
        """Get machine details by ID."""
        machines = self._load_machines()
        return machines.get(f"machine_{machine_id}")
    
    def get_all_machines(self) -> List[Dict[str, Any]]:
        """Get all machines."""
        machines = self._load_machines()
        return list(machines.values())
    
    def record_prediction(self, machine_id: int, prediction_data: Dict[str, Any]) -> bool:
        """
        Record a prediction for a machine.
        
        Args:
            machine_id: ID of the machine
            prediction_data: Prediction data including risk, probability, etc.
            
        Returns:
            True if successful, False otherwise
        """
        try:
            machine_file = self.predictions_dir / f"machine_{machine_id}_predictions.json"
            
            # Load existing predictions
            predictions = []
            if machine_file.exists():
                with open(machine_file, 'r') as f:
                    predictions = json.load(f)
            
            # Add new prediction
            prediction_data['timestamp'] = datetime.now().isoformat()
            predictions.append(prediction_data)
            
            # Keep only last 1000 predictions
            predictions = predictions[-1000:]
            
            # Save
            with open(machine_file, 'w') as f:
                json.dump(predictions, f)
            
            return True
        except Exception as e:
            logger.error(f"Error recording prediction for machine {machine_id}: {e}")
            return False
    
    def get_machine_predictions(self, machine_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent predictions for a machine."""
        try:
            machine_file = self.predictions_dir / f"machine_{machine_id}_predictions.json"
            
            if machine_file.exists():
                with open(machine_file, 'r') as f:
                    predictions = json.load(f)
                return predictions[-limit:]
            
            return []
        except Exception as e:
            logger.error(f"Error retrieving predictions for machine {machine_id}: {e}")
            return []
    
    def get_machine_stats(self, machine_id: int) -> Dict[str, Any]:
        """Get statistics for a machine."""
        try:
            predictions = self.get_machine_predictions(machine_id, limit=500)
            
            if not predictions:
                return {}
            
            # Calculate statistics
            failure_probs = [p.get('failure_probability', 0) for p in predictions]
            risk_levels = [p.get('risk_level', 'UNKNOWN') for p in predictions]
            
            risk_count = {
                'CRITICAL': sum(1 for r in risk_levels if r == 'CRITICAL'),
                'HIGH': sum(1 for r in risk_levels if r == 'HIGH'),
                'MEDIUM': sum(1 for r in risk_levels if r == 'MEDIUM'),
                'LOW': sum(1 for r in risk_levels if r == 'LOW'),
            }
            
            return {
                "machine_id": machine_id,
                "total_predictions": len(predictions),
                "average_failure_probability": round(np.mean(failure_probs), 4),
                "max_failure_probability": round(np.max(failure_probs), 4),
                "min_failure_probability": round(np.min(failure_probs), 4),
                "std_failure_probability": round(np.std(failure_probs), 4),
                "risk_distribution": risk_count,
                "latest_prediction": predictions[-1] if predictions else {},
                "trend": "improving" if np.mean(failure_probs[-50:]) < np.mean(failure_probs[:50]) else "declining"
            }
        except Exception as e:
            logger.error(f"Error calculating stats for machine {machine_id}: {e}")
            return {}
    
    def update_machine_maintenance(self, machine_id: int, scheduled_date: str, maintenance_type: str) -> bool:
        """Update machine maintenance schedule."""
        try:
            machines = self._load_machines()
            key = f"machine_{machine_id}"
            
            if key in machines:
                machines[key]['next_scheduled_maintenance'] = {
                    'date': scheduled_date,
                    'type': maintenance_type,
                    'scheduled_at': datetime.now().isoformat()
                }
                self._save_machines(machines)
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error updating maintenance for machine {machine_id}: {e}")
            return False
    
    def update_machine_status(self, machine_id: int, status: str) -> bool:
        """Update machine operational status."""
        try:
            machines = self._load_machines()
            key = f"machine_{machine_id}"
            
            if key in machines:
                machines[key]['status'] = status
                self._save_machines(machines)
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error updating status for machine {machine_id}: {e}")
            return False
    
    def get_fleet_overview(self) -> Dict[str, Any]:
        """Get overview of entire fleet."""
        try:
            machines = self.get_all_machines()
            
            # Count statuses
            status_counts = {}
            for machine in machines:
                status = machine.get('status', 'UNKNOWN')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Get average metrics
            total_operating_hours = sum(m.get('total_operating_hours', 0) for m in machines)
            total_failures = sum(m.get('total_failures', 0) for m in machines)
            
            return {
                "total_machines": len(machines),
                "status_distribution": status_counts,
                "total_operating_hours": total_operating_hours,
                "total_failures_recorded": total_failures,
                "average_failures_per_machine": round(total_failures / len(machines), 2) if machines else 0,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating fleet overview: {e}")
            return {}
    
    def export_machine_report(self, machine_id: int, output_file: Optional[str] = None) -> str:
        """Export detailed report for a machine."""
        try:
            machine = self.get_machine(machine_id)
            predictions = self.get_machine_predictions(machine_id, limit=200)
            stats = self.get_machine_stats(machine_id)
            
            if not output_file:
                output_file = f"models/machine_{machine_id}_report.json"
            
            report = {
                "machine": machine,
                "statistics": stats,
                "recent_predictions": predictions,
                "generated_at": datetime.now().isoformat()
            }
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            return output_file
        except Exception as e:
            logger.error(f"Error exporting report for machine {machine_id}: {e}")
            return ""
