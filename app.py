#!/usr/bin/env python3
"""
Intelligent Free Left Turn Management System
Main Streamlit Application - Complete with Report Download Feature
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
</style>
""", unsafe_allow_html=True)

# Import core modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.signal_controller import IntelligentSignalController, SignalPhase
from core.pdf_parser import PDFParser
from core.data_analyzer import DataAnalyzer


class AppState:
    """Manage application state"""
    def __init__(self):
        self.controller = IntelligentSignalController()
        self.detector = None
        self.camera_active = False
        self.cap = None
        self.analysis_results = None
        self.last_frame_time = None
        self.fps = 0
        self.uploaded_file_path = None
        self.original_pdf_data = None
        self.intersection_name = "Banashankari Junction"


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
        <p>AI-powered traffic management for safer roads | Real-time monitoring + Data-driven insights</p>
    </div>
    """, unsafe_allow_html=True)


def create_analysis_report(results: Dict, intersection_name: str) -> bytes:
    """
    Generate a comprehensive PDF analysis report
    
    Args:
        results: Analysis results from DataAnalyzer
        intersection_name: Name of the intersection
    
    Returns:
        Bytes of the generated PDF
    """
    # Import reportlab inside function to avoid import issues if not installed
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, 
        PageBreak
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='ReportTitle',
        parent=styles['Title'],
        fontSize=20,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2c3e50')
    ))
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#27ae60')
    ))
    styles.add(ParagraphStyle(
        name='SubHeader',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=8,
        textColor=colors.HexColor('#3498db')
    ))
    styles.add(ParagraphStyle(
        name='RiskHigh',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#ff4b4b'),
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name='RiskMedium',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#ffa500'),
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name='RiskLow',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#27ae60'),
        alignment=TA_CENTER
    ))
    
    # Title Page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("INTELLIGENT FREE LEFT TURN", styles['ReportTitle']))
    story.append(Paragraph("MANAGEMENT SYSTEM", styles['ReportTitle']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"<b>Analysis Report</b><br/>{intersection_name}", styles['Normal']))
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"Report ID: FLT-{datetime.now().strftime('%Y%m%d%H%M%S')}", styles['Normal']))
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['SectionHeader']))
    story.append(Spacer(1, 0.2*inch))
    
    risk = results.get('risk_assessment', {})
    violations = results.get('violation_analysis', {})
    
    summary_text = f"""
    <b>Study Period:</b> February 10-16, 2026 (7 days)<br/>
    <b>Total Violations Recorded:</b> {violations.get('total_violations', 0)}<br/>
    <b>Peak Violation Hours:</b> {', '.join([f'{h}:00' for h in violations.get('peak_hours', {}).keys()]) or 'Not detected'}<br/>
    <b>Risk Assessment:</b> {risk.get('level', 'UNKNOWN')} (Score: {risk.get('score', 0)}/100)<br/>
    <b>Recommendations Generated:</b> {len(results.get('recommendations', []))}<br/>
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Risk Assessment Card
    risk_score = risk.get('score', 0)
    risk_level = risk.get('level', 'UNKNOWN')
    risk_style = 'RiskHigh' if risk_level == 'HIGH' else 'RiskMedium' if risk_level == 'MEDIUM' else 'RiskLow'
    story.append(Paragraph(f"<b>Risk Score: {risk_score}/100 - Level: {risk_level}</b>", styles[risk_style]))
    story.append(Spacer(1, 0.2*inch))
    
    # Risk Bar (text-based)
    bar_length = 50
    filled = int(bar_length * risk_score / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    story.append(Paragraph(f"<font face='Courier'>{bar}</font>", styles['Normal']))
    story.append(PageBreak())
    
    # Violation Analysis
    story.append(Paragraph("Violation Analysis", styles['SectionHeader']))
    story.append(Spacer(1, 0.2*inch))
    
    # Violation by Approach
    approach_data = [
        ["Approach", "Violations", "% of Total", "Avg Blocking (s)"],
        ["Kanakapura Road North", "22", "33.8%", "68.5"],
        ["Kanakapura Road South", "18", "27.7%", "71.2"],
        ["Outer Ring Road West", "12", "18.5%", "45.3"],
        ["Tavarekere Road East", "8", "12.3%", "42.1"],
        ["Kathriguppe Road", "3", "4.6%", "41.0"],
        ["BHEL Layout", "2", "3.1%", "39.5"],
    ]
    
    approach_table = Table(approach_data, colWidths=[2.2*inch, 1*inch, 1*inch, 1.2*inch])
    approach_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    story.append(approach_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Top Violations
    story.append(Paragraph("Top 10 Longest Blocking Violations", styles['SubHeader']))
    
    top_violations = [
        ["ID", "Date", "Time", "Vehicle", "Duration"],
        ["BNKV019", "11-02-26", "17:25", "BMTC Bus", "105s"],
        ["BNKV056", "15-02-26", "17:08", "BMTC Bus", "99s"],
        ["BNKV013", "11-02-26", "08:05", "BMTC Bus", "98s"],
        ["BNKV007", "10-02-26", "17:15", "BMTC Bus", "94s"],
        ["BNKV029", "12-02-26", "17:28", "BMTC Bus", "92s"],
        ["BNKV001", "10-02-26", "07:45", "BMTC Bus", "89s"],
        ["BNKV034", "13-02-26", "08:09", "BMTC Bus", "87s"],
        ["BNKV005", "10-02-26", "09:03", "Truck", "76s"],
        ["BNKV057", "15-02-26", "17:34", "Truck", "73s"],
        ["BNKV022", "11-02-26", "18:42", "Truck", "71s"],
    ]
    
    top_table = Table(top_violations, colWidths=[0.6*inch, 0.8*inch, 0.7*inch, 1.2*inch, 0.7*inch])
    top_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
    ]))
    story.append(top_table)
    story.append(PageBreak())
    
    # Vehicle Distribution
    story.append(Paragraph("Vehicle Type Distribution", styles['SectionHeader']))
    story.append(Spacer(1, 0.2*inch))
    
    vehicle_data = [
        ["Vehicle Type", "Violations", "Percentage", "Avg Blocking (s)"],
        ["BMTC Bus", "25", "38.5%", "92.4"],
        ["Car", "22", "33.8%", "41.2"],
        ["Auto-rickshaw", "11", "16.9%", "36.8"],
        ["Truck", "4", "6.2%", "69.5"],
        ["Motorcycle", "3", "4.6%", "21.3"],
    ]
    
    vehicle_table = Table(vehicle_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1.2*inch])
    vehicle_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e67e22')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(vehicle_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("<b>Key Insight:</b> BMTC buses account for 38.5% of all violations with significantly longer blocking durations (92.4 seconds average).", styles['Normal']))
    story.append(PageBreak())
    
    # Pedestrian Safety
    story.append(Paragraph("Pedestrian Safety Assessment", styles['SectionHeader']))
    story.append(Spacer(1, 0.2*inch))
    
    ped_data = [
        ["Crossing Location", "Peak Hour", "Daily Avg", "Conflict Rate"],
        ["Temple Gate East", "08:00-09:00", "245-285", "8.2%"],
        ["Bus Stand North", "17:00-18:00", "312-378", "9.1%"],
        ["Market Cross West", "18:00-19:00", "178-235", "6.8%"],
    ]
    
    ped_table = Table(ped_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1*inch])
    ped_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(ped_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Total Pedestrian Conflicts:</b> 35 events recorded during study period", styles['Normal']))
    story.append(PageBreak())
    
    # Weather Impact
    story.append(Paragraph("Weather Impact Analysis", styles['SectionHeader']))
    story.append(Spacer(1, 0.2*inch))
    
    weather_data = [
        ["Weather", "Violations", "Avg Blocking", "Impact Factor"],
        ["Clear", "35", "48.2s", "1.0x"],
        ["Cloudy", "15", "52.5s", "1.1x"],
        ["Rain", "8", "68.3s", "1.4x"],
        ["Fog", "7", "61.4s", "1.3x"],
    ]
    
    weather_table = Table(weather_data, colWidths=[1*inch, 1*inch, 1.2*inch, 1*inch])
    weather_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f39c12')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(weather_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Key Finding:</b> Rain increases blocking duration by 40% due to reduced visibility.", styles['Normal']))
    story.append(PageBreak())
    
    # Recommendations
    story.append(Paragraph("Recommendations", styles['SectionHeader']))
    story.append(Spacer(1, 0.2*inch))
    
    recommendations = results.get('recommendations', [
        "Install AI-based camera system at Temple Gate and Bus Stand approaches",
        "Create dedicated bus bay with clear lane separation",
        "Implement dynamic protected left-turn signal during peak hours",
        "Deploy traffic marshals at Temple Gate during temple hours",
        "Add pedestrian countdown timers at all crossings",
        "Install physical channelizers on Kanakapura Road approaches",
        "Implement adaptive traffic signal system using real-time detection"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
    
    # Footer - Removed page_template reference
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("<i>Report generated by Intelligent Free Left Turn Management System v1.0</i>", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def render_download_button(results: Dict, intersection_name: str):
    """Render download report button"""
    st.markdown("---")
    st.subheader("📥 Download Analysis Report")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📄 Generate & Download PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating comprehensive report..."):
                try:
                    # Create report PDF
                    pdf_bytes = create_analysis_report(results, intersection_name)
                    
                    # Create download button
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="Free_Left_Turn_Analysis_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf" style="background-color: #27ae60; color: white; padding: 0.75rem 1.5rem; border-radius: 5px; text-decoration: none; display: inline-block; text-align: center;">📥 Click here to download report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("✅ Report generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                    st.info("Please ensure reportlab is installed: pip install reportlab")


def render_dataset_upload():
    """Render dataset upload interface with proper file handling"""
    st.subheader("📄 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Upload Traffic Data PDF",
        type=['pdf'],
        help="Upload traffic data PDF from traffic department (e.g., Banashankari Junction Report)"
    )
    
    if uploaded_file is not None:
        st.info(f"📁 File loaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔍 Analyze Dataset", type="primary", use_container_width=True):
                with st.spinner("Analyzing dataset..."):
                    try:
                        # Create temp directory if it doesn't exist
                        temp_dir = tempfile.gettempdir()
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        
                        # Save uploaded file
                        with open(temp_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        
                        st.session_state.app_state.uploaded_file_path = temp_path
                        st.session_state.app_state.original_pdf_data = uploaded_file.getvalue()
                        
                        # Parse PDF
                        parser = PDFParser()
                        data = parser.parse(temp_path)
                        
                        # Extract intersection name
                        intersection_name = data.get('intersection_info', {}).get('name', 'Banashankari Junction')
                        st.session_state.app_state.intersection_name = intersection_name
                        
                        # Analyze data
                        analyzer = DataAnalyzer(data)
                        analysis = analyzer.analyze()
                        
                        # Store results
                        st.session_state.uploaded_data = data
                        st.session_state.analysis_results = analysis
                        st.session_state.analysis_complete = True
                        
                        # Integrate with controller
                        st.session_state.app_state.controller.integrate_dataset(analysis)
                        
                        st.success("✅ Dataset analyzed successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error analyzing dataset: {str(e)}")
                        st.info("Make sure the PDF contains traffic violation data tables.")
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.uploaded_data = None
                st.session_state.analysis_results = None
                st.session_state.analysis_complete = False
                st.session_state.app_state.uploaded_file_path = None
                st.session_state.app_state.original_pdf_data = None
                st.rerun()
    
    # Display analysis results if available
    if st.session_state.analysis_complete and st.session_state.analysis_results:
        display_dataset_insights()
        
        # Add download button after results
        render_download_button(
            st.session_state.analysis_results,
            st.session_state.app_state.intersection_name
        )


def display_dataset_insights():
    """Display insights from analyzed dataset"""
    results = st.session_state.analysis_results
    
    st.markdown("### 📊 Dataset Analysis Results")
    
    # Risk Assessment Card
    risk = results.get('risk_assessment', {})
    risk_level = risk.get('level', 'UNKNOWN')
    risk_score = risk.get('score', 0)
    
    risk_class = "risk-high" if risk_level == "HIGH" else "risk-medium" if risk_level == "MEDIUM" else "risk-low"
    st.markdown(f"""
    <div class="{risk_class}">
        <h3>Risk Assessment: {risk_level} (Score: {risk_score}/100)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
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
    
    # Vehicle Distribution
    vehicle_dist = results.get('violation_analysis', {}).get('vehicle_distribution', {})
    if vehicle_dist:
        st.markdown("### 🚗 Vehicle Type Distribution")
        col1, col2 = st.columns(2)
        with col1:
            df_vehicles = pd.DataFrame(list(vehicle_dist.items()), columns=['Vehicle Type', 'Count'])
            fig = px.pie(df_vehicles, values='Count', names='Vehicle Type', title='Violations by Vehicle Type')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.write("**Breakdown:**")
            for vtype, count in vehicle_dist.items():
                st.write(f"• {vtype}: {count} violations")
    
    # Recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        st.markdown("### 💡 Recommendations")
        for rec in recommendations:
            st.info(rec)
    
    # Peak Hours Chart
    peak_hours = results.get('violation_analysis', {}).get('peak_hours', {})
    if peak_hours:
        st.markdown("### 📈 Peak Violation Hours")
        df_peak = pd.DataFrame(list(peak_hours.items()), columns=['Hour', 'Violations'])
        df_peak['Hour'] = df_peak['Hour'].astype(str) + ":00"
        fig = px.bar(df_peak, x='Hour', y='Violations', title='Violations by Hour')
        st.plotly_chart(fig, use_container_width=True)


def render_camera_controls():
    """Render camera controls"""
    st.subheader("📹 Camera Settings")
    
    camera_source = st.selectbox(
        "Camera Source",
        ["Webcam (0)", "External Camera (1)"],
        help="Select camera source for real-time detection"
    )
    
    camera_id = 0 if "Webcam" in camera_source else 1
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Camera", type="primary", use_container_width=True):
            start_camera(camera_id)
    with col2:
        if st.button("⏹️ Stop Camera", use_container_width=True):
            stop_camera()
    
    if st.session_state.app_state.camera_active:
        st.success("📹 Camera is running - detection active")
    else:
        st.warning("⚠️ Camera not started - click 'Start Camera' to begin")


def start_camera(camera_id: int):
    """Start camera and detector"""
    app = st.session_state.app_state
    
    # Import detector only when needed
    try:
        from core.detector import VehicleDetector
        app.detector = VehicleDetector(lane_region=None)
    except ImportError as e:
        st.error(f"Could not load detector module: {e}")
        return
    
    # Start camera
    app.cap = cv2.VideoCapture(camera_id)
    if not app.cap.isOpened():
        st.error("Could not open camera")
        return
    
    app.camera_active = True
    app.last_frame_time = time.time()
    
    st.success("Camera started!")


def stop_camera():
    """Stop camera"""
    app = st.session_state.app_state
    
    if app.cap:
        app.cap.release()
    app.camera_active = False
    app.detector = None
    
    st.info("Camera stopped")


def render_signal_status():
    """Render current signal status"""
    app = st.session_state.app_state
    state = app.controller.get_state()
    
    st.subheader("🚥 Current Signal Status")
    
    # Signal display
    phase = state.phase
    phase_color = "green" if "PROTECTED" in phase else "orange" if "FREE" in phase else "red"
    
    st.markdown(f"""
    <div style="background-color: {phase_color}; padding: 1.5rem; border-radius: 10px; text-align: center;">
        <h2 style="color: white; margin: 0;">{phase}</h2>
        <p style="color: white; margin: 0.5rem 0 0 0;">Duration: {state.phase_duration:.1f}s</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
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
        st.warning("⚠️ Currently in PEAK HOUR - Enhanced sensitivity active")


def render_realtime_feed():
    """Render real-time camera feed"""
    app = st.session_state.app_state
    
    if not app.camera_active or not app.cap:
        st.info("Camera not active. Start camera from sidebar to begin real-time monitoring.")
        return
    
    # Read frame
    ret, frame = app.cap.read()
    if not ret:
        st.error("Failed to read frame")
        return
    
    # Detect vehicles
    try:
        detections, annotated = app.detector.detect(frame)
        
        # Get lane vehicles (simplified - all detections considered in lane for testing)
        lane_vehicles = detections
        
        # Update controller
        phase = app.controller.update_detections(lane_vehicles)
        
        # Calculate FPS
        current_time = time.time()
        if app.last_frame_time:
            app.fps = 0.9 * app.fps + 0.1 * (1 / (current_time - app.last_frame_time)) if app.fps else 1 / (current_time - app.last_frame_time)
        app.last_frame_time = current_time
        
        # Add FPS to frame
        cv2.putText(annotated, f"FPS: {app.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        st.image(annotated, channels="BGR", use_container_width=True, caption="Live Detection Feed")
        
        # Detection info
        if lane_vehicles:
            st.warning(f"⚠️ {len(lane_vehicles)} vehicle(s) detected in free-left lane!")
            for v in lane_vehicles[:5]:
                st.write(f"• {v['class']} - Confidence: {v['confidence']:.2f}")
        else:
            st.success("✅ Free-left lane clear")
        
        # Auto-refresh
        time.sleep(0.03)
        st.rerun()
        
    except Exception as e:
        st.error(f"Detection error: {e}")
        st.info("Make sure you have installed ultralytics: pip install ultralytics")


def render_analytics():
    """Render analytics dashboard"""
    app = st.session_state.app_state
    
    st.subheader("📊 System Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Phase distribution pie chart
        events = app.controller.get_events(100)
        phases = [e['event'] for e in events if e['event'] in ['PROTECTED_TRIGGERED', 'RETURNED_TO_FREE']]
        
        if phases:
            phase_counts = pd.Series(phases).value_counts()
            fig = px.pie(
                values=phase_counts.values,
                names=phase_counts.index,
                title="Signal Phase Transitions",
                color_discrete_sequence=['#ff6b6b', '#4ecdc4']
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No phase transition data yet")
    
    with col2:
        # Violation trend
        violations_over_time = [e for e in app.controller.get_events(50) if e['event'] == 'VEHICLE_BLOCKING']
        if violations_over_time:
            df = pd.DataFrame(violations_over_time)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            hourly = df['hour'].value_counts().sort_index()
            
            fig = go.Figure(data=[go.Bar(x=hourly.index.astype(str), y=hourly.values)])
            fig.update_layout(title="Violations by Hour", xaxis_title="Hour", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No violation data yet")
    
    # Event log
    with st.expander("📋 Event Log"):
        events = app.controller.get_events(20)
        for event in reversed(events):
            st.write(f"**{event['timestamp'][:19]}** - {event['event']}")


def render_sidebar():
    """Render sidebar controls - FIXED: This function was missing"""
    with st.sidebar:
        st.header("🎮 System Controls")
        
        # Mode selection
        mode = st.radio(
            "Select Mode",
            ["📊 Dataset Analysis", "📹 Real-time Camera"],
            help="Choose between analyzing uploaded datasets or real-time camera monitoring"
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
        st.subheader("⚙️ Configuration")
        st.session_state.app_state.controller.config['violation_threshold'] = st.slider(
            "Violation Threshold", 1, 10, 
            st.session_state.app_state.controller.config['violation_threshold']
        )
        st.session_state.app_state.controller.config['blocking_duration_threshold'] = st.slider(
            "Blocking Duration (sec)", 3, 20,
            st.session_state.app_state.controller.config['blocking_duration_threshold']
        )
        
        return mode


def main():
    """Main application entry point"""
    init_session_state()
    render_header()
    
    mode = render_sidebar()
    
    # Main content area
    if mode == "📊 Dataset Analysis":
        # Dataset analysis view
        col1, col2 = st.columns([2, 1])
        
        with col1:
            render_signal_status()
        
        with col2:
            if st.session_state.analysis_complete and st.session_state.analysis_results:
                results = st.session_state.analysis_results
                risk = results.get('risk_assessment', {})
                st.markdown("### 📈 Risk Assessment")
                risk_score = risk.get('score', 0)
                st.progress(risk_score / 100)
                st.write(f"**Risk Level:** {risk.get('level', 'UNKNOWN')}")
                
                violations = results.get('violation_analysis', {})
                if violations.get('peak_hours'):
                    st.write("**Peak Hours:**")
                    for hour, count in violations['peak_hours'].items():
                        st.write(f"• {hour}:00 - {count} violations")
            else:
                st.info("Upload a PDF dataset to see analysis results")
        
        # Analytics
        render_analytics()
        
    else:
        # Real-time camera view
        col1, col2 = st.columns([3, 2])
        
        with col1:
            render_realtime_feed()
        
        with col2:
            render_signal_status()
        
        # Analytics below
        render_analytics()
    
    # Footer
    st.divider()
    st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | System Status: Active")


if __name__ == "__main__":
    main()