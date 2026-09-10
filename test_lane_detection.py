"""
Quick test to verify lane region detection
"""

import cv2
import numpy as np
from core.detector import VehicleDetector

# Define lane region (adjust these coordinates for your video)
LANE_REGION = [
    (100, 400),   # top-left
    (500, 400),   # top-right  
    (550, 450),   # bottom-right
    (50, 450)     # bottom-left
]

def test_with_video(video_path):
    """Test lane detection on a video"""
    
    # Initialize detector with lane region
    detector = VehicleDetector(lane_region=LANE_REGION)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vehicles
        detections, annotated = detector.detect(frame)
        
        # Get vehicles in lane
        lane_vehicles = detector.get_lane_vehicles(detections)
        
        # Get blocking vehicles (stationary for >3 seconds)
        blocking = detector.get_blocking_vehicles(min_duration=3.0)
        
        # Print results every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {len(detections)} vehicles, "
                  f"{len(lane_vehicles)} in lane, {len(blocking)} blocking")
        
        # Display
        cv2.imshow('Detection', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_with_video("your_video.mp4")