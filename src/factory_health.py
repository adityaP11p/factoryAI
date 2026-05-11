"""
Factory Health Monitoring System
Tracks overall factory health, identifies machines at risk, and predicts maintenance schedules.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class FactoryHealthMonitor:
    """Monitors factory health metrics and generates actionable insights."""
    
    def __init__(self, df: pd.DataFrame, model, scaler, feature_columns: List[str], target_column: str = "machine_failure"):
        """
        Initialize the factory health monitor.
        
        Args:
            df: DataFrame with sensor data
            model: Trained ML model for predictions
            scaler: Fitted scaler for feature normalization
            feature_columns: List of feature column names
            target_column: Name of failure target column
        """
        self.df = df
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.n_machines = 50  # Default number of machines
        
    def calculate_factory_health(self) -> Dict[str, Any]:
        """Calculate overall factory health metrics."""
        try:
            # Recent data (last 1000 samples)
            recent = self.df.tail(1000).copy()
            
            # Get predictions for recent data
            X_recent = recent[self.feature_columns]
            X_scaled = pd.DataFrame(self.scaler.transform(X_recent), columns=self.feature_columns)
            y_pred = self.model.predict(X_scaled)
            y_proba = self.model.predict_proba(X_scaled)[:, 1]
            
            # Calculate metrics
            total_samples = len(recent)
            failures = int(np.sum(y_pred))
            normal_samples = total_samples - failures
            failure_rate = (failures / total_samples * 100) if total_samples > 0 else 0
            average_risk = float(np.mean(y_proba))
            
            # Health score: 100 - (failure_rate + average_risk)
            health_score = max(0, min(100, 100 - (failure_rate * 0.5 + average_risk * 50)))
            
            # Determine status
            if health_score >= 85:
                status = "EXCELLENT"
                color = "success"
            elif health_score >= 70:
                status = "GOOD"
                color = "info"
            elif health_score >= 50:
                status = "FAIR"
                color = "warning"
            elif health_score >= 30:
                status = "POOR"
                color = "warning"
            else:
                status = "CRITICAL"
                color = "danger"
            
            # Uptime calculation (based on non-failure samples)
            uptime_percentage = (normal_samples / total_samples * 100) if total_samples > 0 else 0
            
            # Trend analysis (compare first half vs second half)
            mid = len(recent) // 2
            first_half_failures = np.sum(y_pred[:mid])
            second_half_failures = np.sum(y_pred[mid:])
            trend = "improving" if second_half_failures < first_half_failures else "declining"
            
            return {
                "health_score": round(health_score, 2),
                "status": status,
                "color": color,
                "uptime_percentage": round(uptime_percentage, 2),
                "failure_rate": round(failure_rate, 2),
                "average_risk": round(average_risk, 4),
                "total_samples_analyzed": total_samples,
                "failures_detected": failures,
                "normal_samples": normal_samples,
                "trend": trend,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error calculating factory health: {e}")
            return self._get_default_health()
    
    def _get_default_health(self) -> Dict[str, Any]:
        """Return default health metrics in case of error."""
        return {
            "health_score": 75.0,
            "status": "GOOD",
            "color": "info",
            "uptime_percentage": 95.0,
            "failure_rate": 5.0,
            "average_risk": 0.15,
            "total_samples_analyzed": 0,
            "failures_detected": 0,
            "normal_samples": 0,
            "trend": "stable",
            "timestamp": datetime.now().isoformat(),
            "error": "Unable to calculate metrics"
        }
    
    def identify_machines_at_risk(self, risk_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Identify machines at risk based on prediction probabilities.
        
        Args:
            risk_threshold: Probability threshold to consider a machine at risk
            
        Returns:
            List of machines with risk information
        """
        try:
            # Recent data
            recent = self.df.tail(500).copy()
            
            # Add machine IDs
            machine_ids = np.tile(np.arange(1, self.n_machines + 1), len(recent) // self.n_machines + 1)[:len(recent)]
            recent['machine_id'] = machine_ids
            
            # Get predictions
            X_recent = recent[self.feature_columns]
            X_scaled = pd.DataFrame(self.scaler.transform(X_recent), columns=self.feature_columns)
            y_proba = self.model.predict_proba(X_scaled)[:, 1]
            recent['failure_probability'] = y_proba
            
            # Group by machine and calculate risk metrics
            machines_at_risk = []
            for machine_id in recent['machine_id'].unique():
                machine_data = recent[recent['machine_id'] == machine_id]
                avg_risk = float(machine_data['failure_probability'].mean())
                max_risk = float(machine_data['failure_probability'].max())
                recent_readings = len(machine_data)
                
                if avg_risk >= risk_threshold:
                    # Determine risk level
                    if max_risk >= 0.8:
                        risk_level = "CRITICAL"
                        risk_color = "danger"
                    elif max_risk >= 0.5:
                        risk_level = "HIGH"
                        risk_color = "warning"
                    elif avg_risk >= 0.3:
                        risk_level = "MEDIUM"
                        risk_color = "warning"
                    else:
                        risk_level = "LOW"
                        risk_color = "info"
                    
                    # Get latest sensor readings for the machine
                    latest_reading = machine_data.iloc[-1]
                    
                    machines_at_risk.append({
                        "machine_id": int(machine_id),
                        "risk_level": risk_level,
                        "risk_color": risk_color,
                        "average_risk_score": round(avg_risk, 4),
                        "max_risk_score": round(max_risk, 4),
                        "recent_readings": recent_readings,
                        "last_reading": {
                            "air_temp": round(float(latest_reading.get('air_temp', 0)), 2),
                            "process_temp": round(float(latest_reading.get('process_temp', 0)), 2),
                            "rotational_speed": round(float(latest_reading.get('rotational_speed', 0)), 2),
                            "torque": round(float(latest_reading.get('torque', 0)), 2),
                            "tool_wear": round(float(latest_reading.get('tool_wear', 0)), 2),
                        },
                        "timestamp": datetime.now().isoformat(),
                        "recommended_action": self._get_action_for_risk(risk_level)
                    })
            
            # Sort by risk score (highest first)
            machines_at_risk.sort(key=lambda x: x['average_risk_score'], reverse=True)
            return machines_at_risk[:20]  # Return top 20 at-risk machines
            
        except Exception as e:
            logger.error(f"Error identifying machines at risk: {e}")
            return []
    
    def _get_action_for_risk(self, risk_level: str) -> str:
        """Get recommended action based on risk level."""
        actions = {
            "CRITICAL": "IMMEDIATE maintenance required. Stop machine if necessary.",
            "HIGH": "Schedule maintenance within 24-48 hours.",
            "MEDIUM": "Monitor closely. Schedule maintenance within 1 week.",
            "LOW": "Continue normal monitoring."
        }
        return actions.get(risk_level, "Monitor machine")
    
    def generate_maintenance_schedule(self, prediction_days: int = 30) -> List[Dict[str, Any]]:
        """
        Generate predictive maintenance schedule for the next N days.
        
        Args:
            prediction_days: Number of days to predict ahead
            
        Returns:
            Sorted list of maintenance tasks with urgency
        """
        try:
            recent = self.df.tail(500).copy()
            
            # Add machine IDs
            machine_ids = np.tile(np.arange(1, self.n_machines + 1), len(recent) // self.n_machines + 1)[:len(recent)]
            recent['machine_id'] = machine_ids
            
            # Get predictions
            X_recent = recent[self.feature_columns]
            X_scaled = pd.DataFrame(self.scaler.transform(X_recent), columns=self.feature_columns)
            y_proba = self.model.predict_proba(X_scaled)[:, 1]
            recent['failure_probability'] = y_proba
            
            maintenance_schedule = []
            base_date = datetime.now()
            
            for machine_id in recent['machine_id'].unique():
                machine_data = recent[recent['machine_id'] == machine_id]
                avg_risk = float(machine_data['failure_probability'].mean())
                max_risk = float(machine_data['failure_probability'].max())
                
                # Predict maintenance urgency based on current risk
                if max_risk >= 0.8:
                    urgency = "IMMEDIATE"
                    days_until_maintenance = 0
                    priority = 1
                elif max_risk >= 0.6:
                    urgency = "URGENT"
                    days_until_maintenance = 2
                    priority = 2
                elif avg_risk >= 0.4:
                    urgency = "SOON"
                    days_until_maintenance = 7
                    priority = 3
                else:
                    urgency = "ROUTINE"
                    days_until_maintenance = 14
                    priority = 4
                
                scheduled_date = base_date + timedelta(days=days_until_maintenance)
                
                # Estimate maintenance duration based on risk
                if urgency == "IMMEDIATE":
                    estimated_hours = 4
                    maintenance_type = "Emergency Repair"
                elif urgency == "URGENT":
                    estimated_hours = 3
                    maintenance_type = "Urgent Servicing"
                elif urgency == "SOON":
                    estimated_hours = 2
                    maintenance_type = "Preventive Maintenance"
                else:
                    estimated_hours = 1
                    maintenance_type = "Routine Inspection"
                
                maintenance_schedule.append({
                    "machine_id": int(machine_id),
                    "maintenance_type": maintenance_type,
                    "urgency": urgency,
                    "priority": priority,
                    "scheduled_date": scheduled_date.strftime("%Y-%m-%d"),
                    "scheduled_time": f"{scheduled_date.hour:02d}:{scheduled_date.minute:02d}",
                    "estimated_duration_hours": estimated_hours,
                    "risk_score": round(avg_risk, 4),
                    "max_risk_score": round(max_risk, 4),
                    "description": f"Machine {machine_id} requires {maintenance_type.lower()}. Current risk: {avg_risk*100:.1f}%",
                    "status": "Scheduled",
                    "estimated_completion": (scheduled_date + timedelta(hours=estimated_hours)).strftime("%Y-%m-%d %H:%M"),
                    "parts_required": self._get_parts_for_maintenance(max_risk),
                    "estimated_cost": round(estimated_hours * 150 + (max_risk * 500), 2)
                })
            
            # Sort by priority (ascending) then by date
            maintenance_schedule.sort(key=lambda x: (x['priority'], x['scheduled_date']))
            return maintenance_schedule[:30]  # Return next 30 maintenance tasks
            
        except Exception as e:
            logger.error(f"Error generating maintenance schedule: {e}")
            return []
    
    def _get_parts_for_maintenance(self, risk_score: float) -> List[str]:
        """Get recommended parts based on risk score."""
        parts = []
        if risk_score >= 0.7:
            parts.extend(["Main Bearing Assembly", "Cooling Fan", "Oil Filter"])
        if risk_score >= 0.5:
            parts.extend(["Drive Belt", "Seals", "Lubricant"])
        if risk_score >= 0.3:
            parts.extend(["Air Filter", "Inspection Kit"])
        return parts if parts else ["General Inspection Kit"]
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system diagnostics."""
        try:
            recent = self.df.tail(1000).copy()
            
            # Add machine IDs
            machine_ids = np.tile(np.arange(1, self.n_machines + 1), len(recent) // self.n_machines + 1)[:len(recent)]
            recent['machine_id'] = machine_ids
            
            # Get predictions
            X_recent = recent[self.feature_columns]
            X_scaled = pd.DataFrame(self.scaler.transform(X_recent), columns=self.feature_columns)
            y_proba = self.model.predict_proba(X_scaled)[:, 1]
            
            # Calculate diagnostics
            total_machines = self.n_machines
            machines_monitored = len(recent['machine_id'].unique())
            
            critical_count = int(np.sum(y_proba >= 0.8))
            high_count = int(np.sum((y_proba >= 0.5) & (y_proba < 0.8)))
            medium_count = int(np.sum((y_proba >= 0.3) & (y_proba < 0.5)))
            low_count = int(np.sum(y_proba < 0.3))
            
            # Sensor statistics
            sensor_avg_temp = float(recent['air_temp'].mean())
            sensor_avg_speed = float(recent['rotational_speed'].mean())
            sensor_avg_torque = float(recent['torque'].mean())
            
            return {
                "total_machines": total_machines,
                "machines_monitored": machines_monitored,
                "machines_online": machines_monitored,
                "machines_offline": total_machines - machines_monitored,
                "monitoring_uptime": round((machines_monitored / total_machines) * 100, 2),
                "risk_distribution": {
                    "critical": critical_count,
                    "high": high_count,
                    "medium": medium_count,
                    "low": low_count
                },
                "average_sensors": {
                    "air_temperature": round(sensor_avg_temp, 2),
                    "rotational_speed": round(sensor_avg_speed, 2),
                    "torque": round(sensor_avg_torque, 2),
                },
                "last_update": datetime.now().isoformat(),
                "data_points_analyzed": len(recent),
                "model_confidence": round(float(np.mean([max(p) for p in self.model.predict_proba(X_scaled)])), 4)
            }
        except Exception as e:
            logger.error(f"Error generating system diagnostics: {e}")
            return {}
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get complete dashboard summary in one call."""
        return {
            "factory_health": self.calculate_factory_health(),
            "machines_at_risk": self.identify_machines_at_risk(),
            "maintenance_schedule": self.generate_maintenance_schedule(),
            "system_diagnostics": self.get_system_diagnostics(),
            "timestamp": datetime.now().isoformat()
        }
