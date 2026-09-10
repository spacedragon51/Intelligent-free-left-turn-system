#!/usr/bin/env python3
"""
Intelligent Free Left Turn Management System
Main Streamlit Application - Supports Camera & Video Upload
"""

import streamlit as st
import cv2
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os
import tempfile
import json
import io
import base64
from typing import Optional, Dict, Any
import logging

# Page config
st.set_page_config(
    page_title="Intelligent Free Left Turn Management",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-selector {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #ff4b4b;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .risk-medium {
        background-color: #ffa500;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .risk-low {
        background-color: #00cc66;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .video-upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Import core modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.signal_controller import IntelligentSignalController, SignalPhase
from core.pdf_parser import PDFParser
from core.data_analyzer import DataAnalyzer
from core.detector import VehicleDetector, VideoSource


class AppState:
    """Manage application state"""
    def __init__(self):
        self.controller = IntelligentSignalController()
        self.detector = None
        self.video_source = None
        self.camera_active = False
        self.analysis_results = None
        self.last_frame_time = None
        self.fps = 0
        self.uploaded_file_path = None
        self.original_pdf_data = None
        self.intersection_name = "Banashankari Junction"
        self.current_mode = "camera"  # camera or video
        self.video_progress = 0
        self.uploaded_video_path = None


def init_session_state():
    """Initialize session state variables"""
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None


def render_header():
    """Render main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚦 Intelligent Free Left Turn Management System</h1>
        <p>AI-powered traffic management | Real-time monitoring | Dataset Analysis | Video Testing</p>
    </div>
    """, unsafe_allow_html=True)


def render_camera_controls():
    """Render camera/video controls - FIXED: Shows video upload clearly"""
    """Render camera/video controls with lane configuration"""
    st.subheader("📹 Source Selection")
    
    # ========== ADD LANE CONFIGURATION UI ==========
    st.subheader("🎯 Free-Left Lane Configuration")
    
    # Default lane coordinates (adjust based on typical video)
    col1, col2 = st.columns(2)
    with col1:
        lane_x1 = st.number_input("Top-Left X", 0, 1280, 100, key="lane_x1")
        lane_y1 = st.number_input("Top-Left Y", 0, 720, 400, key="lane_y1")
    with col2:
        lane_x2 = st.number_input("Bottom-Right X", 0, 1280, 550, key="lane_x2")
        lane_y2 = st.number_input("Bottom-Right Y", 0, 720, 450, key="lane_y2")
    
    # Create lane region from user input
    user_lane_region = [
        (lane_x1, lane_y1),
        (lane_x2, lane_y1),
        (lane_x2, lane_y2),
        (lane_x1, lane_y2)
    ]
    
    # Show preview of lane region
    import numpy as np
    preview = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.polylines(preview, [np.array(user_lane_region, dtype=np.int32)], True, (0, 255, 255), 2)
    cv2.putText(preview, "FREE LEFT LANE", (lane_x1, lane_y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    st.image(preview, caption="Lane Region Preview", use_container_width=True)
    
    # Store in session state
    if 'lane_region' not in st.session_state:
        st.session_state.lane_region = user_lane_region
    # =============================================
    
    # Rest of your camera controls (tabs for camera/video)
    
    # Create tabs for different sources
    tab1, tab2 = st.tabs(["📷 Live Camera", "🎬 Video File Upload"])
    
    source_info = None
    
    with tab1:
        st.markdown("### Live Camera")
        st.info("Use your webcam for real-time detection")
        
        camera_id = st.selectbox(
            "Select Camera Device",
            ["Webcam (0)", "External Camera (1)", "USB Camera (2)"],
            index=0,
            key="camera_select"
        )
        camera_index = int(camera_id.split("(")[1].split(")")[0])
        
        source_info = {
            'type': 'camera',
            'source': camera_index
        }
        
        st.success("✅ Camera selected. Click 'Start' below to begin.")
    
    with tab2:
        st.markdown("### Video File Upload")
        st.markdown('<div class="video-upload-area">', unsafe_allow_html=True)
        st.markdown("🎥 **Upload a pre-recorded video file**")
        st.markdown("Supported formats: MP4, AVI, MOV, MKV, WEBM")
        
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Upload traffic video for testing",
            key="video_uploader"
        )
        
        if uploaded_video is not None:
            # Save uploaded video to temp file
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"uploaded_video_{int(time.time())}.mp4")
            
            with open(temp_path, 'wb') as f:
                f.write(uploaded_video.getbuffer())
            
            st.session_state.app_state.uploaded_video_path = temp_path
            
            # Display video info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Name", uploaded_video.name[:30] + "..." if len(uploaded_video.name) > 30 else uploaded_video.name)
            with col2:
                st.metric("File Size", f"{uploaded_video.size / 1024:.1f} KB")
            with col3:
                # Get video duration
                cap = cv2.VideoCapture(temp_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                st.metric("Duration", f"{duration:.1f} sec")
            
            source_info = {
                'type': 'video',
                'source': temp_path,
                'name': uploaded_video.name,
                'duration': duration
            }
            
            st.success("✅ Video loaded successfully!")
        else:
            st.info("📁 No video file selected. Upload a video to begin testing.")
            st.markdown("💡 **Tip:** You can generate a test video using the provided script.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Control buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ START", type="primary", use_container_width=True):
            if source_info:
                if start_video_source(source_info):
                    st.rerun()
            else:
                st.warning("Please select a source first (camera or video)")
    
    with col2:
        if st.button("⏹️ STOP", use_container_width=True):
            stop_video_source()
            st.rerun()
    
    # Show current status
    st.markdown("---")
    st.subheader("📊 Status")
    
    if st.session_state.app_state.camera_active:
        if st.session_state.app_state.video_source:
            info = st.session_state.app_state.video_source.get_info()
            st.success(f"✅ **ACTIVE:** {info['type'].upper()} source running")
            if info['type'] == 'video':
                st.info(f"🎬 Playing: {st.session_state.app_state.uploaded_video_path}")
        else:
            st.success("✅ Source is active")
    else:
        st.warning("⚠️ No active source - select camera or video and click START")


def start_video_source(source_info):
    """Start video source (camera or video file)"""
    app = st.session_state.app_state
    
    if source_info is None:
        st.warning("Please select a video source first")
        return False
    
    try:
        # ========== ADD LANE REGION HERE ==========
        # Define the free-left lane region (adjust coordinates based on your video)
        # These coordinates work for a 1280x720 video
        lane_region = [
            (100, 400),   # top-left corner
            (500, 400),   # top-right corner
            (550, 450),   # bottom-right corner
            (50, 450)     # bottom-left corner
        ]
        
        # For videos with different resolution, scale coordinates
        # If you know the video dimensions, uncomment this:
        # if source_info['type'] == 'video':
        #     cap = cv2.VideoCapture(source_info['source'])
        #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #     cap.release()
        #     
        #     # Scale coordinates to match video resolution
        #     lane_region = [
        #         (int(100 * width / 1280), int(400 * height / 720)),
        #         (int(500 * width / 1280), int(400 * height / 720)),
        #         (int(550 * width / 1280), int(450 * height / 720)),
        #         (int(50 * width / 1280), int(450 * height / 720))
        #     ]
        # ==========================================
        
        # Initialize detector WITH lane region
        app.detector = VehicleDetector(lane_region=lane_region)  # <-- PASS IT HERE
        
        # Create video source
        app.video_source = VideoSource(source_info['source'])
        
        if not app.video_source.open():
            st.error(f"Could not open source: {source_info['source']}")
            return False
        
        app.camera_active = True
        app.last_frame_time = time.time()
        
        # Display source info
        source_info_data = app.video_source.get_info()
        if source_info_data['type'] == 'video':
            st.success(f"✅ Video loaded: {source_info.get('name', 'video')}")
            st.info(f"📹 Lane region configured: {lane_region}")
        else:
            st.success("✅ Camera started successfully!")
        
        return True
        
    except Exception as e:
        st.error(f"Error starting source: {str(e)}")
        return False


def stop_video_source():
    """Stop video source"""
    app = st.session_state.app_state
    
    if app.video_source:
        app.video_source.release()
    
    app.camera_active = False
    app.detector = None
    app.video_source = None
    
    st.info("Source stopped")


def render_realtime_feed():
    """Render real-time feed from camera or video"""
    app = st.session_state.app_state
    
    if not app.camera_active or not app.video_source:
        st.info("📹 No active source. Select camera or upload video from the sidebar and click START.")
        return
    
    # Read frame
    ret, frame = app.video_source.read()
    
    if not ret:
        st.warning("End of video reached or no frame available")
        source_info = app.video_source.get_info()
        if source_info['type'] == 'video':
            st.info("Video playback completed. Click START again to replay.")
            # Reset video to beginning
            if app.video_source:
                app.video_source.reset()
        stop_video_source()
        return
    
    # Detect vehicles
    try:
        detections, annotated = app.detector.detect(frame)
        
        # Get lane vehicles
        lane_vehicles = app.detector.get_lane_vehicles(detections)
        
        # Get blocking vehicles
        blocking = app.detector.get_blocking_vehicles(min_duration=3.0)
        
        # Update controller with lane vehicles
        phase = app.controller.update_detections(lane_vehicles)
        
        # Calculate FPS
        current_time = time.time()
        if app.last_frame_time:
            app.fps = 0.9 * app.fps + 0.1 * (1 / (current_time - app.last_frame_time)) if app.fps else 1 / (current_time - app.last_frame_time)
        app.last_frame_time = current_time
        
        # Add overlays to frame
        cv2.putText(annotated, f"FPS: {app.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add source info
        source_info = app.video_source.get_info()
        cv2.putText(annotated, f"Source: {source_info['type'].upper()}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add progress for video
        if source_info['type'] == 'video':
            progress = app.video_source.get_progress()
            bar_width = int(annotated.shape[1] * 0.8)
            bar_height = 5
            bar_x = annotated.shape[1] // 2 - bar_width // 2
            bar_y = annotated.shape[0] - 20
            
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
            cv2.putText(annotated, f"Progress: {progress*100:.1f}%", (bar_x, bar_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        st.image(annotated, channels="BGR", use_container_width=True, caption="Live Detection Feed")
        
        # Detection info
        col1, col2 = st.columns(2)
        
        with col1:
            if lane_vehicles:
                st.warning(f"⚠️ {len(lane_vehicles)} vehicle(s) in free-left lane!")
                for v in lane_vehicles[:3]:
                    st.write(f"• {v['class']} - Conf: {v['confidence']:.2f}")
            else:
                st.success("✅ Free-left lane clear")
        
        with col2:
            if blocking:
                st.error(f"🚨 {len(blocking)} vehicle(s) BLOCKING free-left lane!")
                for v in blocking[:3]:
                    st.write(f"• Vehicle {v['id'][:8]} - {v['duration']:.1f}s")
            else:
                st.info("No blocking vehicles detected")
        
        # Auto-refresh for continuous playback
        time.sleep(0.03)
        st.rerun()
        
    except Exception as e:
        st.error(f"Detection error: {e}")
        st.info("Make sure you have installed: pip install ultralytics opencv-python")


def render_sidebar():
    """Render sidebar controls"""
    with st.sidebar:
        st.header("🎮 System Controls")
        
        # Mode selection
        mode = st.radio(
            "Select Mode",
            ["📊 Dataset Analysis", "📹 Real-time Camera/Video"],
            help="Choose between analyzing uploaded datasets or real-time camera/video monitoring"
        )
        
        st.divider()
        
        if mode == "📊 Dataset Analysis":
            render_dataset_upload()
        else:
            render_camera_controls()
        
        st.divider()
        
        # Manual override controls
        st.subheader("🔧 Manual Override")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔒 Force Protected Left", type="primary", use_container_width=True):
                st.session_state.app_state.controller.manual_protect()
                st.success("Protected left activated")
        with col2:
            if st.button("🔄 Reset to Auto", use_container_width=True):
                st.session_state.app_state.controller.manual_reset()
                st.info("Auto mode restored")
        
        st.divider()
        
        # Configuration
        st.subheader("⚙️ Detection Settings")
        st.session_state.app_state.controller.config['violation_threshold'] = st.slider(
            "Violation Threshold", 1, 10, 
            st.session_state.app_state.controller.config['violation_threshold']
        )
        
        return mode


def render_dataset_upload():
    """Render dataset upload interface"""
    st.subheader("📄 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Upload Traffic Data PDF",
        type=['pdf'],
        help="Upload traffic data PDF from traffic department"
    )
    
    if uploaded_file is not None:
        st.info(f"📁 File loaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        
        if st.button("🔍 Analyze Dataset", type="primary", use_container_width=True):
            with st.spinner("Analyzing dataset..."):
                try:
                    temp_dir = tempfile.gettempdir()
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    parser = PDFParser()
                    data = parser.parse(temp_path)
                    
                    intersection_name = data.get('intersection_info', {}).get('name', 'Banashankari Junction')
                    st.session_state.app_state.intersection_name = intersection_name
                    
                    analyzer = DataAnalyzer(data)
                    analysis = analyzer.analyze()
                    
                    st.session_state.uploaded_data = data
                    st.session_state.analysis_results = analysis
                    st.session_state.analysis_complete = True
                    
                    st.session_state.app_state.controller.integrate_dataset(analysis)
                    
                    st.success("✅ Dataset analyzed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    if st.session_state.analysis_complete and st.session_state.analysis_results:
        display_dataset_insights()
        render_download_button(
            st.session_state.analysis_results,
            st.session_state.app_state.intersection_name
        )


def display_dataset_insights():
    """Display insights from analyzed dataset"""
    results = st.session_state.analysis_results
    
    st.markdown("### 📊 Dataset Analysis Results")
    
    risk = results.get('risk_assessment', {})
    risk_level = risk.get('level', 'UNKNOWN')
    risk_score = risk.get('score', 0)
    
    risk_class = "risk-high" if risk_level == "HIGH" else "risk-medium" if risk_level == "MEDIUM" else "risk-low"
    st.markdown(f"""
    <div class="{risk_class}">
        <h3>Risk Assessment: {risk_level} (Score: {risk_score}/100)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Violations", results.get('violation_analysis', {}).get('total_violations', 0))
    with col2:
        st.metric("Traffic Records", results.get('statistics', {}).get('traffic_volume_records', 0))
    with col3:
        st.metric("Pedestrian Records", results.get('statistics', {}).get('pedestrian_records', 0))
    with col4:
        peak_hours = results.get('violation_analysis', {}).get('peak_hours', {})
        if peak_hours:
            peak_str = ', '.join([f"{h}:00" for h in peak_hours.keys()])
            st.metric("Peak Hours", peak_str)
    
    recommendations = results.get('recommendations', [])
    if recommendations:
        st.markdown("### 💡 Recommendations")
        for rec in recommendations[:5]:
            st.info(rec)


def render_download_button(results: Dict, intersection_name: str):
    """Render download report button"""
    st.markdown("---")
    st.subheader("📥 Download Analysis Report")
    
    if st.button("📄 Generate & Download PDF Report", type="primary", use_container_width=True):
        with st.spinner("Generating report..."):
            try:
                from reportlab.lib import colors
                from reportlab.lib.pagesizes import A4
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.lib.units import inch
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                story = []
                
                styles = getSampleStyleSheet()
                
                story.append(Spacer(1, 2*inch))
                story.append(Paragraph("Intelligent Free Left Turn Management", styles['Title']))
                story.append(Paragraph(f"Analysis Report: {intersection_name}", styles['Normal']))
                story.append(Spacer(1, 1*inch))
                story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
                
                doc.build(story)
                buffer.seek(0)
                
                b64 = base64.b64encode(buffer.getvalue()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="Free_Left_Turn_Report_{datetime.now().strftime("%Y%m%d")}.pdf" style="background-color: #27ae60; color: white; padding: 0.75rem 1.5rem; border-radius: 5px; text-decoration: none;">📥 Click to download report</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("Report generated!")
                
            except Exception as e:
                st.error(f"Error: {e}")


def render_signal_status():
    """Render current signal status"""
    app = st.session_state.app_state
    state = app.controller.get_state()
    
    st.subheader("🚥 Current Signal Status")
    
    phase = state.phase
    phase_color = "green" if "PROTECTED" in phase else "orange"
    
    st.markdown(f"""
    <div style="background-color: {phase_color}; padding: 1.5rem; border-radius: 10px; text-align: center;">
        <h2 style="color: white; margin: 0;">{phase}</h2>
        <p style="color: white; margin: 0.5rem 0 0 0;">Duration: {state.phase_duration:.1f}s</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🚗 Blocking Vehicles", state.blocking_vehicles)
    with col2:
        st.metric("⚠️ Violations", state.violations_detected)
    with col3:
        st.metric("⏱️ Cooldown", f"{state.cooldown_remaining:.0f}s")
    with col4:
        st.metric("📊 Risk Level", state.risk_level)
    
    if state.is_peak_hour:
        st.warning("⚠️ Peak Hour - Enhanced sensitivity")


def render_analytics():
    """Render analytics dashboard"""
    app = st.session_state.app_state
    
    st.subheader("📊 System Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        events = app.controller.get_events(100)
        phases = [e['event'] for e in events if e['event'] in ['PROTECTED_TRIGGERED', 'RETURNED_TO_FREE']]
        
        if phases:
            phase_counts = pd.Series(phases).value_counts()
            fig = px.pie(values=phase_counts.values, names=phase_counts.index, title="Signal Phase Transitions")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        violations = [e for e in events if e['event'] == 'VEHICLE_BLOCKING']
        if violations:
            df = pd.DataFrame(violations)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            hourly = df['hour'].value_counts().sort_index()
            
            fig = go.Figure(data=[go.Bar(x=hourly.index.astype(str), y=hourly.values)])
            fig.update_layout(title="Violations by Hour")
            st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📋 Event Log"):
        for event in reversed(events[-20:]):
            st.write(f"**{event['timestamp'][:19]}** - {event['event']}")

# Add to app.py
class MultiCameraSystem:
    def __init__(self):
        self.cameras = {
            'north_approach': {'id': 0, 'lane_region': north_lane}, # type: ignore
            'south_approach': {'id': 1, 'lane_region': south_lane}, # pyright: ignore[reportUndefinedVariable]
            'east_approach': {'id': 2, 'lane_region': east_lane}, # pyright: ignore[reportUndefinedVariable]
            'west_approach': {'id': 3, 'lane_region': west_lane} # pyright: ignore[reportUndefinedVariable]
        }
    
    def monitor_all(self):
        results = {}
        for name, cam in self.cameras.items():
            results[name] = self.detect(cam) # pyright: ignore[reportAttributeAccessIssue]
        return results


def main():
    """Main application entry point"""
    init_session_state()
    render_header()
    
    mode = render_sidebar()
    
    if mode == "📊 Dataset Analysis":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            render_signal_status()
        
        with col2:
            if st.session_state.analysis_complete:
                results = st.session_state.analysis_results
                risk = results.get('risk_assessment', {})
                st.progress(risk.get('score', 0) / 100)
                st.write(f"**Risk Level:** {risk.get('level', 'UNKNOWN')}")
        
        render_analytics()
        
    else:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            render_realtime_feed()
        
        with col2:
            render_signal_status()
        
        render_analytics()
    
    st.divider()
    st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | System Status: Active")


if __name__ == "__main__":
    main()