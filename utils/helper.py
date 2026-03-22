"""
Utility helper functions
"""

import json
import os
from datetime import datetime
from typing import Dict, Any


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save analysis results to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def load_results(filepath: str) -> Dict[str, Any]:
    """Load analysis results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_duration(seconds: float) -> str:
    """Format seconds into readable duration"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def get_timestamp() -> str:
    """Get formatted timestamp"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")