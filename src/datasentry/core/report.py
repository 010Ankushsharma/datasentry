"""Report generation for DataSentry library.

This module provides comprehensive report generation capabilities
for data quality detection results.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from datasentry.core.base import DetectionResult, SeverityLevel


class ReportGenerator:
    """Generate comprehensive reports from detection results.
    
    This class aggregates results from multiple detectors and generates
    detailed reports in various formats (dict, JSON, HTML).
    
    Attributes:
        results: List of DetectionResult objects.
        metadata: Additional metadata for the report.
    
    Example:
        >>> results = [imbalance_result, outlier_result, missing_result]
        >>> report_gen = ReportGenerator(results)
        >>> report = report_gen.generate_summary()
        >>> html_report = report_gen.to_html()
    """
    
    def __init__(
        self,
        results: List[DetectionResult],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize the report generator.
        
        Args:
            results: List of DetectionResult objects from detectors.
            metadata: Optional metadata about the dataset or analysis.
        """
        self.results = results
        self.metadata = metadata or {}
        self._timestamp = datetime.now()
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary report of all detection results.
        
        Returns:
            Dictionary containing comprehensive summary.
        """
        total_detectors = len(self.results)
        issues_detected = sum(1 for r in self.results if r.issue_detected)
        
        # Count by severity
        severity_counts = {level.name: 0 for level in SeverityLevel}
        for result in self.results:
            severity_counts[result.severity.name] += 1
        
        # Filter to only show detected issues
        detected_results = [r for r in self.results if r.issue_detected]
        
        # Calculate overall health score
        health_score = self._calculate_health_score()
        
        summary = {
            "report_metadata": {
                "generated_at": self._timestamp.isoformat(),
                "total_detectors": total_detectors,
                "issues_detected": issues_detected,
                "health_score": health_score,
                "overall_status": self._get_overall_status(health_score),
            },
            "severity_distribution": severity_counts,
            "detailed_results": [r.to_dict() for r in detected_results],
            "all_results": [r.to_dict() for r in self.results],
            "recommendations": self._aggregate_recommendations(),
            "dataset_metadata": self.metadata,
        }
        
        return summary
    
    def _calculate_health_score(self) -> float:
        """Calculate overall data health score.
        
        Returns:
            Health score between 0.0 (worst) and 1.0 (best).
        """
        if not self.results:
            return 1.0
        
        # Weight by severity
        severity_weights = {
            SeverityLevel.NONE: 1.0,
            SeverityLevel.LOW: 0.8,
            SeverityLevel.MEDIUM: 0.5,
            SeverityLevel.HIGH: 0.2,
            SeverityLevel.CRITICAL: 0.0,
        }
        
        total_score = sum(
            severity_weights[r.severity] for r in self.results
        )
        return round(total_score / len(self.results), 3)
    
    def _get_overall_status(self, health_score: float) -> str:
        """Get overall status based on health score.
        
        Args:
            health_score: Calculated health score.
        
        Returns:
            Status string.
        """
        if health_score >= 0.9:
            return "EXCELLENT"
        elif health_score >= 0.7:
            return "GOOD"
        elif health_score >= 0.5:
            return "FAIR"
        elif health_score >= 0.3:
            return "POOR"
        else:
            return "CRITICAL"
    
    def _aggregate_recommendations(self) -> List[Dict[str, Any]]:
        """Aggregate and prioritize all recommendations.
        
        Returns:
            List of prioritized recommendations.
        """
        recommendations = []
        
        for result in self.results:
            if result.issue_detected and result.recommendations:
                for rec in result.recommendations:
                    recommendations.append({
                        "detector": result.detector_name,
                        "severity": result.severity.name,
                        "recommendation": rec,
                        "priority": self._get_priority(result.severity),
                    })
        
        # Sort by priority (lower number = higher priority)
        recommendations.sort(key=lambda x: x["priority"])
        
        return recommendations
    
    def _get_priority(self, severity: SeverityLevel) -> int:
        """Get numeric priority from severity.
        
        Args:
            severity: Severity level.
        
        Returns:
            Priority number (1-5, 1 being highest).
        """
        priority_map = {
            SeverityLevel.CRITICAL: 1,
            SeverityLevel.HIGH: 2,
            SeverityLevel.MEDIUM: 3,
            SeverityLevel.LOW: 4,
            SeverityLevel.NONE: 5,
        }
        return priority_map.get(severity, 5)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary.
        
        Returns:
            Dictionary representation of the report.
        """
        return self.generate_summary()
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string.
        
        Args:
            indent: Indentation level for pretty printing.
        
        Returns:
            JSON string representation of the report.
        """
        summary = self.generate_summary()
        return json.dumps(summary, indent=indent, default=str)
    
    def to_html(self) -> str:
        """Generate HTML report.
        
        Returns:
            HTML string representation of the report.
        """
        summary = self.generate_summary()
        meta = summary["report_metadata"]
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DataSentry Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }}
                .summary-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px 20px 10px 0;
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #667eea;
                }}
                .metric-label {{
                    color: #666;
                    font-size: 0.9em;
                }}
                .status-{meta['overall_status'].lower()} {{
                    color: {'#28a745' if meta['overall_status'] == 'EXCELLENT' else '#ffc107' if meta['overall_status'] in ['GOOD', 'FAIR'] else '#dc3545'};
                    font-weight: bold;
                }}
                .severity-critical {{ color: #dc3545; }}
                .severity-high {{ color: #fd7e14; }}
                .severity-medium {{ color: #ffc107; }}
                .severity-low {{ color: #17a2b8; }}
                .severity-none {{ color: #28a745; }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #667eea;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .recommendation {{
                    background: #e3f2fd;
                    padding: 10px;
                    margin: 5px 0;
                    border-left: 4px solid #2196f3;
                    border-radius: 4px;
                }}
                .health-score {{
                    font-size: 3em;
                    font-weight: bold;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>DataSentry Quality Report</h1>
                <p>Generated: {meta['generated_at']}</p>
            </div>
            
            <div class="summary-card">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <div class="metric-value status-{meta['overall_status'].lower()}">{meta['overall_status']}</div>
                    <div class="metric-label">Overall Status</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{meta['health_score']:.1%}</div>
                    <div class="metric-label">Health Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{meta['issues_detected']}</div>
                    <div class="metric-label">Issues Detected</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{meta['total_detectors']}</div>
                    <div class="metric-label">Checks Performed</div>
                </div>
            </div>
            
            <div class="summary-card">
                <h2>Severity Distribution</h2>
                <table>
                    <tr>
                        <th>Severity</th>
                        <th>Count</th>
                    </tr>
        """
        
        for severity, count in summary["severity_distribution"].items():
            html += f"""
                    <tr>
                        <td class="severity-{severity.lower()}">{severity}</td>
                        <td>{count}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        """
        
        # Issues table
        if summary["detailed_results"]:
            html += """
            <div class="summary-card">
                <h2>Detected Issues</h2>
                <table>
                    <tr>
                        <th>Detector</th>
                        <th>Severity</th>
                        <th>Score</th>
                        <th>Details</th>
                    </tr>
            """
            
            for result in summary["detailed_results"]:
                details_str = "<br>".join(
                    f"{k}: {v}" for k, v in result["details"].items()
                )
                html += f"""
                    <tr>
                        <td>{result['detector_name']}</td>
                        <td class="severity-{result['severity'].lower()}">{result['severity']}</td>
                        <td>{result['score']:.3f}</td>
                        <td>{details_str}</td>
                    </tr>
                """
            
            html += """
                </table>
            </div>
            """
        
        # Recommendations
        if summary["recommendations"]:
            html += """
            <div class="summary-card">
                <h2>Recommendations</h2>
            """
            
            for rec in summary["recommendations"][:10]:  # Show top 10
                html += f"""
                <div class="recommendation">
                    <strong>[{rec['severity']}] {rec['detector']}:</strong> {rec['recommendation']}
                </div>
                """
            
            html += """
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def save_html(self, filepath: str) -> None:
        """Save HTML report to file.
        
        Args:
            filepath: Path to save the HTML file.
        """
        html = self.to_html()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
    
    def save_json(self, filepath: str, indent: int = 2) -> None:
        """Save JSON report to file.
        
        Args:
            filepath: Path to save the JSON file.
            indent: Indentation level for pretty printing.
        """
        json_str = self.to_json(indent=indent)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_str)
    
    def get_critical_issues(self) -> List[DetectionResult]:
        """Get all critical and high severity issues.
        
        Returns:
            List of critical/high severity DetectionResults.
        """
        return [
            r for r in self.results 
            if r.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH)
        ]
    
    def get_issues_by_severity(
        self, 
        severity: SeverityLevel
    ) -> List[DetectionResult]:
        """Get all issues of a specific severity.
        
        Args:
            severity: Severity level to filter by.
        
        Returns:
            List of DetectionResults with specified severity.
        """
        return [r for r in self.results if r.severity == severity]
