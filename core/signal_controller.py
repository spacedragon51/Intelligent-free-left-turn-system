"""
Intelligent Signal Controller - Decision Engine
Manages real-time signal decisions based on detection and dataset insights
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalPhase(Enum):
    """Signal phase states"""
    FREE_LEFT = "🟡 FREE LEFT (Yield to oncoming)"
    PROTECTED_LEFT = "🟢 PROTECTED LEFT (Green Arrow)"
    ALL_RED = "🔴 ALL RED (Safety Stop)"
    PEDESTRIAN = "🚶 PEDESTRIAN CROSSING"


@dataclass
class SignalState:
    """Current signal state data"""
    phase: str
    phase_duration: float
    violations_detected: int
    blocking_vehicles: int
    cooldown_remaining: float
    is_peak_hour: bool
    risk_level: str


class IntelligentSignalController:
    """
    Intelligent controller for free-left turn management
    Uses real-time detections and historical dataset insights
    """
    
    def __init__(self):
        """Initialize controller with default configuration"""
        self.config = {
            'violation_threshold': 5,
            'blocking_duration_threshold': 10,
            'min_protected_duration': 15,
            'max_free_left_duration': 120,
            'cooldown_after_protected': 60,
            'dynamic_thresholds': True
        }
        
        self.current_phase = SignalPhase.FREE_LEFT
        self.phase_start_time = time.time()
        self.blocking_vehicles: Dict[str, float] = {}
        self.cooldown_until = 0
        self.manual_override = False
        self.dataset_insights = None
        self.peak_hours = [8, 9, 17, 18]  # Default peak hours
        self.event_log: List[Dict] = []
        
    def integrate_dataset(self, insights: Dict[str, Any]) -> None:
        """
        Integrate dataset insights to adjust thresholds
        
        Args:
            insights: Analysis report from dataset
        """
        risk = insights.get('risk_assessment', {})
        risk_level = risk.get('level', 'MEDIUM')
        violations = insights.get('violation_analysis', {})
        
        # Adjust thresholds based on risk level
        if risk_level == 'HIGH':
            self.config['violation_threshold'] = 3
            self.config['blocking_duration_threshold'] = 5
            self.config['max_free_left_duration'] = 60
            self.config['cooldown_after_protected'] = 45
        elif risk_level == 'MEDIUM':
            self.config['violation_threshold'] = 5
            self.config['blocking_duration_threshold'] = 8
            self.config['max_free_left_duration'] = 90
        else:  # LOW
            self.config['violation_threshold'] = 7
            self.config['blocking_duration_threshold'] = 12
            self.config['max_free_left_duration'] = 150
        
        # Extract peak hours
        peak_hours = violations.get('peak_hours', {})
        if peak_hours:
            self.peak_hours = [int(h) for h in peak_hours.keys()]
        
        self.dataset_insights = insights
        self._log_event('DATASET_INTEGRATED', {
            'risk_level': risk_level,
            'thresholds': self.config
        })
        
        logger.info(f"Dataset integrated - Risk: {risk_level}, Threshold: {self.config['violation_threshold']}")
    
    def update_detections(self, detections: List[Dict]) -> SignalPhase:
        """
        Update with real-time detections and determine next phase
        
        Args:
            detections: List of detected vehicles in free-left lane
            
        Returns:
            Current signal phase
        """
        if self.manual_override:
            return self.current_phase
        
        current_time = time.time()
        
        # Check cooldown
        if current_time < self.cooldown_until:
            return SignalPhase.FREE_LEFT
        
        # Update blocking vehicles
        current_vehicles = {d.get('id', str(i)) for i, d in enumerate(detections)}
        
        # Add new blocking vehicles
        for det in detections:
            vid = det.get('id', str(id(det)))
            if vid not in self.blocking_vehicles:
                self.blocking_vehicles[vid] = current_time
                self._log_event('VEHICLE_BLOCKING', {'vehicle_id': vid})
        
        # Remove resolved vehicles
        resolved = set(self.blocking_vehicles.keys()) - current_vehicles
        for vid in resolved:
            duration = current_time - self.blocking_vehicles[vid]
            self._log_event('VEHICLE_RESOLVED', {'vehicle_id': vid, 'duration': duration})
            del self.blocking_vehicles[vid]
        
        # Count persistent blocks
        persistent = sum(1 for start in self.blocking_vehicles.values() 
                        if current_time - start > self.config['blocking_duration_threshold'])
        
        # Decision logic
        if self.current_phase == SignalPhase.FREE_LEFT:
            # Check if protected left should be triggered
            should_protect = (
                len(detections) >= self.config['violation_threshold'] or
                persistent >= 2 or
                (current_time - self.phase_start_time) > self.config['max_free_left_duration']
            )
            
            # More sensitive during peak hours
            if self.is_peak_hour() and len(detections) >= max(1, self.config['violation_threshold'] - 1):
                should_protect = True
            
            if should_protect:
                self.current_phase = SignalPhase.PROTECTED_LEFT
                self.phase_start_time = current_time
                self._log_event('PROTECTED_TRIGGERED', {
                    'violations': len(detections),
                    'persistent': persistent
                })
                logger.info(f"Protected left triggered - {len(detections)} violations")
        
        elif self.current_phase == SignalPhase.PROTECTED_LEFT:
            # Check if protected phase should end
            phase_duration = current_time - self.phase_start_time
            
            if phase_duration >= self.config['min_protected_duration']:
                if len(detections) == 0 and len(self.blocking_vehicles) == 0:
                    self.current_phase = SignalPhase.FREE_LEFT
                    self.phase_start_time = current_time
                    self.cooldown_until = current_time + self.config['cooldown_after_protected']
                    self._log_event('RETURNED_TO_FREE', {'duration': phase_duration})
                    logger.info("Returned to free-left mode")
        
        return self.current_phase
    
    def is_peak_hour(self) -> bool:
        """Check if current time is within peak hours"""
        current_hour = datetime.now().hour
        return current_hour in self.peak_hours
    
    def get_state(self) -> SignalState:
        """Get current signal state"""
        current_time = time.time()
        return SignalState(
            phase=self.current_phase.value,
            phase_duration=current_time - self.phase_start_time,
            violations_detected=len(self.blocking_vehicles),
            blocking_vehicles=len(self.blocking_vehicles),
            cooldown_remaining=max(0, self.cooldown_until - current_time),
            is_peak_hour=self.is_peak_hour(),
            risk_level=self.dataset_insights.get('risk_assessment', {}).get('level', 'MEDIUM') if self.dataset_insights else 'UNKNOWN'
        )
    
    def manual_protect(self) -> None:
        """Manual override to trigger protected left"""
        self.manual_override = True
        self.current_phase = SignalPhase.PROTECTED_LEFT
        self.phase_start_time = time.time()
        self._log_event('MANUAL_PROTECT', {})
        logger.info("Manual override - Protected left activated")
    
    def manual_reset(self) -> None:
        """Manual reset to automatic mode"""
        self.manual_override = False
        self.current_phase = SignalPhase.FREE_LEFT
        self.phase_start_time = time.time()
        self._log_event('MANUAL_RESET', {})
        logger.info("Manual reset - Automatic mode restored")
    
    def _log_event(self, event_type: str, data: Dict) -> None:
        """Log event for audit trail"""
        self.event_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': event_type,
            'data': data
        })
        
        # Keep last 1000 events
        if len(self.event_log) > 1000:
            self.event_log = self.event_log[-1000:]
    
    def get_events(self, limit: int = 50) -> List[Dict]:
        """Get recent events"""
        return self.event_log[-limit:]