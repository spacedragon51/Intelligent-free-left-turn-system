"""
Real-time Vehicle Detector using YOLOv8
Detects vehicles blocking free-left lane
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)


class VehicleDetector:
    """
    Real-time vehicle detector for free-left lane monitoring
    Uses YOLOv8 for object detection
    """
    
    # COCO class IDs for vehicles
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(self, model_path: str = 'yolov8n.pt', lane_region: Optional[List[Tuple[int, int]]] = None):
        """
        Initialize detector
        
        Args:
            model_path: Path to YOLO model
            lane_region: Polygon coordinates of free-left lane (optional)
        """
        self.model = YOLO(model_path)
        self.lane_region = lane_region
        self.tracker = {}  # Track vehicles across frames
        self.frame_count = 0
        
        logger.info(f"Detector initialized with model: {model_path}")
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect vehicles in frame
        
        Args:
            frame: Input image/frame
            
        Returns:
            Tuple of (detections list, annotated frame)
        """
        self.frame_count += 1
        
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Check if it's a vehicle
                if class_id in self.VEHICLE_CLASSES:
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    detection = {
                        'id': f"v{self.frame_count}_{len(detections)}",
                        'class': self.VEHICLE_CLASSES[class_id],
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'center': center,
                        'in_lane': self._in_lane(center) if self.lane_region else False
                    }
                    detections.append(detection)
        
        # Annotate frame
        annotated = self._annotate(frame, detections)
        
        return detections, annotated
    
    def _in_lane(self, point: Tuple[int, int]) -> bool:
        """Check if point is inside lane region"""
        if not self.lane_region:
            return False
        
        return cv2.pointPolygonTest(
            np.array(self.lane_region, dtype=np.int32),
            point,
            False
        ) >= 0
    
    def _annotate(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Annotate frame with detection results"""
        annotated = frame.copy()
        
        # Draw lane region if defined
        if self.lane_region:
            cv2.polylines(
                annotated,
                [np.array(self.lane_region, dtype=np.int32)],
                True,
                (0, 255, 255),
                2
            )
            cv2.putText(
                annotated,
                "FREE LEFT LANE",
                (self.lane_region[0][0], self.lane_region[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 0, 255) if det['in_lane'] else (0, 255, 0)
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, det['center'], 5, color, -1)
            
            label = f"{det['class']} ({det['confidence']:.2f})"
            if det['in_lane']:
                label += " [BLOCKING]"
            
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        return annotated
    
    def get_lane_vehicles(self, detections: List[Dict]) -> List[Dict]:
        """Get only vehicles in free-left lane"""
        return [d for d in detections if d.get('in_lane', False)]
    
    def set_lane_region(self, region: List[Tuple[int, int]]) -> None:
        """Set lane region polygon"""
        self.lane_region = region
        logger.info(f"Lane region set: {region}")