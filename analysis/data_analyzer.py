"""
Data Analyzer Module for Traffic Dataset Analysis
Generates insights, risk assessment, and recommendations from parsed data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class TrafficDataAnalyzer:
    """
    Analyzes parsed traffic data to generate actionable insights
    Implements methods from intelligent transportation systems literature
    """
    
    def __init__(self, parsed_data: Dict[str, Any]):
        """
        Initialize the analyzer with parsed data
        
        Args:
            parsed_data: Dictionary containing parsed data from PDF
        """
        self.data = parsed_data
        self.insights: Dict[str, Any] = {
            'violation_patterns': {},
            'peak_hours': {},
            'risk_assessment': {},
            'recommendations': [],
            'statistical_summary': {}
        }
    
    def analyze_violations(self) -> Dict[str, Any]:
        """
        Analyze violation patterns from dataset
        
        Returns:
            Dictionary with violation analysis insights
        """
        violations_df = pd.DataFrame(self.data.get('violations', []))
        
        if violations_df.empty:
            return {'error': 'No violation data found in dataset', 'total_violations': 0}
        
        insights: Dict[str, Any] = {
            'total_violations': len(violations_df),
            'unique_vehicles': violations_df.get('vehicle_id', pd.Series()).nunique() if 'vehicle_id' in violations_df.columns else 'N/A',
        }
        
        # Time-based analysis if timestamp columns exist
        time_cols = [c for c in violations_df.columns if 'time' in c.lower() or 'hour' in c.lower()]
        if time_cols:
            try:
                violations_df['hour'] = pd.to_datetime(violations_df[time_cols[0]]).dt.hour
                peak_hours = violations_df['hour'].value_counts().head(3).to_dict()
                insights['peak_hours'] = peak_hours
            except Exception:
                pass
        
        # Vehicle type distribution
        vehicle_cols = [c for c in violations_df.columns if 'type' in c.lower() or 'class' in c.lower()]
        if vehicle_cols:
            insights['vehicle_distribution'] = violations_df[vehicle_cols[0]].value_counts().to_dict()
        
        self.insights['violation_patterns'] = insights
        return insights
    
    def analyze_traffic_volume(self) -> Dict[str, Any]:
        """
        Analyze traffic volume patterns
        
        Returns:
            Dictionary with traffic volume analysis
        """
        volume_df = pd.DataFrame(self.data.get('traffic_volume', []))
        
        if volume_df.empty:
            return {'error': 'No traffic volume data found'}
        
        insights: Dict[str, Any] = {
            'total_vehicles': len(volume_df) if 'count' not in volume_df.columns 
                              else int(volume_df['count'].sum()) if pd.api.types.is_numeric_dtype(volume_df['count']) 
                              else 'N/A',
            'peak_volume_times': {}
        }
        
        # Find peak volume times
        if 'hour' in volume_df.columns:
            if 'count' in volume_df.columns and pd.api.types.is_numeric_dtype(volume_df['count']):
                peak = volume_df.groupby('hour')['count'].sum()
            else:
                peak = volume_df.groupby('hour').size()
            insights['peak_volume_times'] = peak.nlargest(3).to_dict()
        
        self.insights['statistical_summary']['traffic_volume'] = insights
        return insights
    
    def calculate_risk_index(self) -> float:
        """
        Calculate risk index based on multiple factors
        Formula: Risk = (Violations_per_hour * Severity_factor) + (Blocking_duration_factor)
        
        Returns:
            Risk score between 0 and 100
        """
        violations = self.insights.get('violation_patterns', {})
        
        risk_score = 0.0
        
        # Factor 1: Violation frequency
        total_violations = violations.get('total_violations', 0)
        if isinstance(total_violations, int):
            if total_violations > 100:
                risk_score += 40
            elif total_violations > 50:
                risk_score += 25
            elif total_violations > 10:
                risk_score += 10
        
        # Factor 2: Peak hour concentration
        peak_hours = violations.get('peak_hours', {})
        if peak_hours:
            max_peak = max(peak_hours.values()) if peak_hours else 0
            if isinstance(max_peak, (int, float)):
                if max_peak > 20:
                    risk_score += 30
                elif max_peak > 10:
                    risk_score += 15
        
        # Factor 3: Two-wheeler involvement (high risk in Indian context)
        vehicle_dist = violations.get('vehicle_distribution', {})
        two_wheeler_risk = 0
        for k, v in vehicle_dist.items():
            if isinstance(k, str) and ('motor' in k.lower() or 'bike' in k.lower() or '2' in k.lower()):
                two_wheeler_risk += v if isinstance(v, (int, float)) else 0
        if two_wheeler_risk > 0:
            risk_score += min(two_wheeler_risk * 0.5, 30)
        
        # Determine risk level
        final_score = min(risk_score, 100)
        risk_level = 'HIGH' if final_score > 60 else 'MEDIUM' if final_score > 30 else 'LOW'
        
        self.insights['risk_assessment'] = {
            'score': final_score,
            'level': risk_level,
            'factors': {
                'violation_frequency': total_violations,
                'peak_concentration': max(peak_hours.values()) if peak_hours else 0,
                'two_wheeler_risk': two_wheeler_risk
            }
        }
        
        return final_score
    
    def generate_recommendations(self) -> List[str]:
        """
        Generate actionable recommendations based on data analysis
        
        Returns:
            List of recommendation strings
        """
        recommendations: List[str] = []
        risk = self.insights.get('risk_assessment', {})
        violations = self.insights.get('violation_patterns', {})
        
        # Risk-based recommendations
        if risk.get('level') == 'HIGH':
            recommendations.append("🚨 IMMEDIATE ACTION: Implement protected left-turn signal phase during peak hours")
            recommendations.append("📹 Deploy AI-based enforcement cameras at this intersection")
            recommendations.append("🚧 Install physical channelizers to separate free-left lane")
        
        elif risk.get('level') == 'MEDIUM':
            recommendations.append("⚠️ Schedule periodic protected left-turn interventions")
            recommendations.append("📊 Monitor violation patterns for 2 weeks before permanent changes")
        
        # Peak hour recommendations
        peak_hours = violations.get('peak_hours', {})
        if peak_hours:
            peak_times = ', '.join([str(h) for h in peak_hours.keys()])
            recommendations.append(f"🕐 Deploy traffic marshals during peak hours: {peak_times}")
        
        # Vehicle-type specific
        vehicle_dist = violations.get('vehicle_distribution', {})
        if any('motor' in str(v).lower() or 'bike' in str(v).lower() for v in vehicle_dist.keys()):
            recommendations.append("🛵 Implement dedicated two-wheeler waiting zone before free-left lane")
        
        # Pedestrian safety
        if self.data.get('pedestrian_data'):
            recommendations.append("🚶 Add pedestrian crossing signals with countdown timers")
        
        self.insights['recommendations'] = recommendations
        return recommendations
    
    def generate_full_report(self) -> Dict[str, Any]:
        """
        Generate complete analysis report
        
        Returns:
            Dictionary with full analysis report
        """
        self.analyze_violations()
        self.analyze_traffic_volume()
        self.calculate_risk_index()
        self.generate_recommendations()
        
        return {
            'dataset_summary': self.data.get('metadata', {}),
            'violation_analysis': self.insights['violation_patterns'],
            'risk_assessment': self.insights['risk_assessment'],
            'recommendations': self.insights['recommendations'],
            'statistical_summary': self.insights['statistical_summary']
        }


# Example usage
if __name__ == "__main__":
    # Test with sample data
    sample_data = {
        'metadata': {'study_period': 'Feb 2026'},
        'violations': [
            {'vehicle_id': 'V001', 'vehicle_type': 'Car', 'timestamp': '08:15:23'},
            {'vehicle_id': 'V002', 'vehicle_type': 'Truck', 'timestamp': '17:30:45'},
        ],
        'traffic_volume': [],
        'pedestrian_data': []
    }
    
    analyzer = TrafficDataAnalyzer(sample_data)
    report = analyzer.generate_full_report()
    
    print("=" * 60)
    print("Analysis Report")
    print("=" * 60)
    print(f"Risk Level: {report['risk_assessment'].get('level')}")
    print(f"Recommendations: {len(report['recommendations'])}")
    for rec in report['recommendations']:
        print(f"  • {rec}")