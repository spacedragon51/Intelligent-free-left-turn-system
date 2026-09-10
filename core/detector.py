"""
Real-time Vehicle Detector using YOLOv8
Supports both camera and video file input
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from ultralytics import YOLO
import logging
import time

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
        
        try:
            self.model = YOLO(model_path)
            logger.info(f"Detector initialized with model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        self.lane_region = lane_region
        self.tracker = {}  # Track vehicles across frames
        self.frame_count = 0
        self.detection_history = []
        
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect vehicles in frame
        
        Args:
            frame: Input image/frame
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Tuple of (detections list, annotated frame)
        """
        self.frame_count += 1
        
        # Run YOLO detection
        results = self.model(frame, verbose=False, conf=conf_threshold)
        
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
                    
                    # Determine if vehicle is in free-left lane
                    in_lane = self._in_lane(center) if self.lane_region else False
                    
                    # Calculate vehicle area
                    area = (x2 - x1) * (y2 - y1)
                    
                    detection = {
                        'id': f"v{self.frame_count}_{len(detections)}",
                        'class': self.VEHICLE_CLASSES[class_id],
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'center': center,
                        'area': area,
                        'in_lane': in_lane,
                        'timestamp': time.time()
                    }
                    detections.append(detection)
        
        # Update tracking
        self._update_tracking(detections)
        
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
    
    def _update_tracking(self, detections: List[Dict]) -> None:
        """
        Update vehicle tracking for blocking detection
        Simplified tracking using position proximity
        """
        current_centers = {d['center']: d for d in detections}
        
        # Update existing tracked vehicles
        for track_id, track_data in list(self.tracker.items()):
            # Find closest detection
            min_dist = float('inf')
            closest_center = None
            
            for center, det in current_centers.items():
                dist = np.sqrt((track_data['center'][0] - center[0])**2 + 
                              (track_data['center'][1] - center[1])**2)
                if dist < min_dist and dist < 50:  # 50 pixel threshold
                    min_dist = dist
                    closest_center = center
            
            if closest_center:
                # Update existing track
                self.tracker[track_id]['center'] = closest_center
                self.tracker[track_id]['last_seen'] = time.time()
                self.tracker[track_id]['frames_seen'] += 1
                self.tracker[track_id]['in_lane'] = current_centers[closest_center]['in_lane']
                self.tracker[track_id]['frames_missing'] = 0
            else:
                # Mark as possibly gone (will be removed after 30 frames)
                self.tracker[track_id]['frames_missing'] += 1
                if self.tracker[track_id]['frames_missing'] > 30:
                    del self.tracker[track_id]
        
        # Add new detections
        for center, det in current_centers.items():
            found = False
            for track_id, track_data in self.tracker.items():
                if np.sqrt((track_data['center'][0] - center[0])**2 + 
                          (track_data['center'][1] - center[1])**2) < 50:
                    found = True
                    break
            if not found:
                self.tracker[f"track_{len(self.tracker)}"] = {
                    'center': center,
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'frames_seen': 1,
                    'frames_missing': 0,
                    'in_lane': det['in_lane']
                }
    
    def get_blocking_vehicles(self, min_duration: float = 3.0) -> List[Dict]:
        """
        Get vehicles that have been blocking the lane for > min_duration
        
        Args:
            min_duration: Minimum blocking duration in seconds
            
        Returns:
            List of blocking vehicles with duration info
        """
        blocking = []
        current_time = time.time()
        
        for track_id, track_data in self.tracker.items():
            if track_data.get('in_lane', False):
                blocking_duration = current_time - track_data.get('first_seen', current_time)
                if blocking_duration >= min_duration:
                    blocking.append({
                        'id': track_id,
                        'duration': blocking_duration,
                        'first_seen': track_data.get('first_seen', current_time),
                        'center': track_data.get('center', (0, 0))
                    })
        
        return blocking
    
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
                label += " ⚠️"
            
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        # Add blocking vehicles info
        blocking = self.get_blocking_vehicles()
        if blocking:
            y_offset = 60
            cv2.putText(annotated, "⚠️ BLOCKING VEHICLES:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            for i, v in enumerate(blocking[:3]):
                cv2.putText(annotated, f"  Vehicle {v['id'][:8]} - {v['duration']:.1f}s", 
                           (10, y_offset + 25 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return annotated
    
    def get_lane_vehicles(self, detections: List[Dict]) -> List[Dict]:
        """Get only vehicles in free-left lane"""
        return [d for d in detections if d.get('in_lane', False)]
    
    def set_lane_region(self, region: List[Tuple[int, int]]) -> None:
        """Set lane region polygon"""
        self.lane_region = region
        logger.info(f"Lane region set: {region}")
    
    def get_stats(self) -> Dict:
        """Get detection statistics"""
        return {
            'total_frames': self.frame_count,
            'active_tracks': len(self.tracker),
            'blocking_vehicles': len(self.get_blocking_vehicles()),
            'detections_per_second': len(self.detection_history) / max(1, self.frame_count / 30)
        }


class VideoSource:
    """Unified video source handler for camera and video files"""
    
    def __init__(self, source: Union[int, str]):
        """
        Initialize video source
        
        Args:
            source: Camera index (int) or video file path (str)
        """
        self.source = source
        self.cap = None
        self.is_camera = isinstance(source, int)
        self.fps = 0
        self.total_frames = 0
        self.current_frame = 0
        self._is_opened = False
        
    def open(self) -> bool:
        """Open video source"""
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                self._is_opened = False
                return False
            
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_camera else 0
            self._is_opened = True
            return True
            
        except Exception as e:
            logger.error(f"Error opening source: {e}")
            self._is_opened = False
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame"""
        if self.cap is None or not self._is_opened:
            return False, None
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
            return True, frame
        else:
            return False, None
    
    def release(self) -> None:
        """Release video source"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self._is_opened = False
    
    def is_opened(self) -> bool:
        """Check if source is open"""
        return self._is_opened and self.cap is not None
    
    def get_progress(self) -> float:
        """Get playback progress (for video files)"""
        if self.is_camera or self.total_frames == 0:
            return 0.0
        if self.total_frames > 0:
            return min(1.0, self.current_frame / self.total_frames)
        return 0.0
    
    def get_info(self) -> Dict:
        """Get source information"""
        info = {
            'type': 'camera' if self.is_camera else 'video',
            'source': str(self.source),
            'is_open': self.is_opened()
        }
        
        if not self.is_camera and self.total_frames > 0:
            info['fps'] = self.fps
            info['total_frames'] = self.total_frames
            info['duration_seconds'] = self.total_frames / self.fps if self.fps > 0 else 0
        
        return info
    
    def reset(self) -> None:
        """Reset video to beginning (for video files)"""
        if not self.is_camera and self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0


# For testing the detector
if __name__ == "__main__":
    print("=" * 60)
    print("Testing VehicleDetector")
    print("=" * 60)
    
    # Create test image
    test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (100, 350), (200, 450), (0, 255, 0), -1)  # Simulate car
    cv2.putText(test_img, "Test Vehicle", (100, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Initialize detector
    try:
        detector = VehicleDetector()
        print("✅ VehicleDetector initialized successfully")
        
        # Test detection
        detections, annotated = detector.detect(test_img)
        print(f"✅ Detection test passed: {len(detections)} vehicles detected")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure you have installed: pip install ultralytics")
    
    print("=" * 60)