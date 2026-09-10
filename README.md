# Intelligent Free-Left Turn Management System

An AI-powered traffic management system designed to improve free-left turn efficiency, reduce intersection congestion, and enhance road safety through real-time computer vision, traffic data analysis, and adaptive signal decision-making.

## Overview

Free-left turns are commonly used at urban intersections to improve traffic throughput. However, when improperly managed, they can become a source of congestion, lane obstruction, pedestrian conflicts, and traffic violations.

The **Intelligent Free-Left Turn Management System** addresses these challenges by combining:

* Real-time vehicle detection using YOLOv8
* Computer vision-based lane and pedestrian analysis
* Traffic dataset and PDF analysis
* Historical traffic pattern analysis
* Adaptive signal decision-making
* Risk assessment and traffic recommendations
* Interactive analytics and monitoring dashboards

The system analyzes both **real-time traffic conditions** and **historical traffic data** to determine whether a free-left movement should remain active or transition to a controlled signal phase.

## Key Features

### Real-Time Traffic Monitoring

The system uses YOLOv8 and OpenCV to analyze live or recorded traffic footage.

Supported input sources include:

* Live camera feeds
* Uploaded video files
* Recorded traffic footage

The system can identify and monitor:

* Cars
* Motorcycles
* Buses
* Trucks
* Vehicles entering the free-left lane
* Vehicles blocking the free-left movement
* Pedestrians crossing the intersection

Vehicle tracking is used to estimate how long vehicles remain within the free-left region and identify potentially problematic traffic conditions.

### Traffic Dataset Analysis

Traffic reports can be uploaded as PDF documents for automated analysis.

The system extracts relevant traffic information such as:

* Traffic volume
* Traffic violations
* Pedestrian activity
* Peak traffic periods
* Historical traffic patterns

The extracted information is then used to generate:

* Traffic risk assessments
* Peak-hour insights
* Congestion indicators
* Recommended traffic management actions

### Adaptive Signal Control

The signal decision engine dynamically evaluates traffic conditions and determines the appropriate free-left state.

Possible states include:

| State          | Description                                                               |
| -------------- | ------------------------------------------------------------------------- |
| Free Left      | Free-left movement remains active under acceptable traffic conditions     |
| Protected Left | Left-turn movement is controlled using a dedicated signal phase           |
| All Red        | Traffic movement is temporarily stopped when safety conditions require it |

The decision engine considers multiple factors, including:

* Number of detected violations
* Vehicle blocking duration
* Traffic volume
* Peak-hour conditions
* Pedestrian activity
* Historical traffic patterns
* Intersection risk level

### Pedestrian and Safety Analysis

The system incorporates pedestrian-related conditions into its decision-making process.

It can analyze:

* Pedestrian crossings
* Zebra crossing activity
* Pedestrian waiting areas
* Potential conflicts between vehicles and pedestrians

When unsafe pedestrian conditions are detected, the signal controller can prioritize a safer traffic phase.

### Analytics Dashboard

The application provides an interactive dashboard for monitoring traffic conditions and system decisions.

Dashboard information can include:

* Current signal state
* Vehicle detection statistics
* Signal phase transitions
* Traffic violations
* Violation trends by hour
* Vehicle blocking duration
* Risk levels
* Historical traffic insights
* Event logs

### Automated Report Generation

The system can generate reports containing:

* Traffic analysis summary
* Detected violations
* Risk assessment
* Peak-hour analysis
* System decisions
* Recommended improvements

Reports can be exported for further analysis or documentation.

---

## System Architecture

The system follows a modular architecture consisting of computer vision, data analysis, decision-making, and visualization components.

```text
                   Traffic Input
                       |
          +------------+------------+
          |                         |
     Live Camera                Video File
          |                         |
          +------------+------------+
                       |
                YOLOv8 Detection
                       |
              Vehicle & Pedestrian
                   Tracking
                       |
              Traffic Conditions
                       |
        +--------------+--------------+
        |                             |
 Real-Time Analysis            Historical Analysis
        |                             |
        |                       PDF Traffic Reports
        |                             |
        +--------------+--------------+
                       |
              Decision Engine
                       |
        +--------------+--------------+
        |              |              |
     Free Left    Protected Left    All Red
                       |
                Analytics Dashboard
                       |
                Traffic Reports
```

## How It Works

### 1. Real-Time Mode

The user can select either a live camera feed or an uploaded video.

The system then:

1. Captures traffic frames.
2. Detects vehicles using YOLOv8.
3. Tracks detected vehicles across frames.
4. Identifies vehicles entering the free-left region.
5. Measures blocking duration.
6. Detects pedestrian activity.
7. Evaluates traffic conditions.
8. Passes the resulting information to the signal decision engine.
9. Determines the appropriate signal state.

### 2. Dataset Analysis Mode

Traffic reports can be uploaded in PDF format.

The system:

1. Extracts relevant information from the PDF.
2. Processes traffic volume and violation data.
3. Identifies peak traffic periods.
4. Evaluates historical traffic conditions.
5. Calculates an overall traffic risk level.
6. Generates recommendations for intersection management.

### 3. Signal Decision Engine

The decision engine combines real-time observations with historical traffic information.

A transition toward a protected left or safety phase may be triggered when conditions such as the following are detected:

* High violation frequency
* Excessive free-left lane blocking
* Significant congestion
* Peak-hour traffic
* Pedestrian crossing activity
* Elevated historical risk

This allows the system to move beyond fixed signal timing and respond to changing intersection conditions.

---

## Risk Assessment

The system classifies traffic conditions into three risk levels:

| Risk Level | Description                                                             |
| ---------- | ----------------------------------------------------------------------- |
| Low        | Traffic conditions are within acceptable limits                         |
| Medium     | Increasing congestion, violations, or pedestrian conflicts are detected |
| High       | Significant congestion or safety risks require intervention             |

Risk assessment is based on factors such as:

* Total traffic violations
* Vehicle blocking duration
* Traffic volume
* Peak-hour conditions
* Pedestrian activity
* Historical traffic patterns

---

## Project Structure

The current repository is organized as follows:

```text
Intelligent-free-left-turn-system/
│
├── analysis/
│   └── ...
│
├── core/
│   └── ...
│
├── utils/
│   └── ...
│
├── app.py
├── cloud_dashboard.py
├── firebase_config.py
├── main.py
├── pdf_conv.py
├── test_lane_detection.py
├── requirements.txt
└── .gitignore
```

The project is divided into separate modules for application logic, core traffic-processing functionality, utilities, analysis components, dashboard functionality, and testing.

---

## Technology Stack

| Component                       | Technology           |
| ------------------------------- | -------------------- |
| User Interface                  | Streamlit            |
| Programming Language            | Python               |
| Computer Vision                 | OpenCV               |
| Object Detection                | YOLOv8 / Ultralytics |
| Data Processing                 | Pandas               |
| Visualization                   | Plotly               |
| PDF Processing                  | pdfplumber           |
| Authentication / Cloud Services | Firebase             |
| Testing                         | Python               |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/spacedragon51/Intelligent-free-left-turn-system.git
cd Intelligent-free-left-turn-system
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will provide the interactive traffic monitoring and analysis interface.

---

## Use Cases

The system can be applied to:

* Smart city traffic management
* Urban intersection optimization
* Free-left turn monitoring
* Traffic police decision support
* Traffic safety analysis
* Government traffic analytics
* Intelligent transportation systems
* Traffic management research
* Hackathon and academic projects

---

## Future Improvements

Potential extensions include:

* Multi-camera intersection monitoring
* Edge deployment using NVIDIA Jetson or Raspberry Pi
* Direct integration with physical traffic signal controllers
* Predictive traffic-flow modelling
* Real-time cloud-based traffic monitoring
* Advanced pedestrian and vehicle trajectory analysis
* Mobile application for traffic authorities
* Integration with IoT-based traffic sensors
* Machine-learning-based traffic demand prediction
* Automated alerts for high-risk traffic conditions

---

## Project Goals

The primary objective of the project is to demonstrate how computer vision and data-driven decision-making can be applied to a real-world traffic management problem.

Rather than relying exclusively on fixed signal timings, the system aims to provide an adaptive approach that considers **current traffic conditions, historical traffic patterns, vehicle behaviour, and pedestrian safety** when determining how a free-left movement should be managed.

## Repository

**GitHub:** https://github.com/spacedragon51/Intelligent-free-left-turn-system

## License

This project is intended for educational, research, and prototype development purposes.
