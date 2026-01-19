#!/usr/bin/env python3
"""
NeuroSQL Deployment Monitoring
Monitors system health and risk metrics
"""

import time
import json
import psutil
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List
import logging
from prometheus_client import start_http_server, Gauge, Counter, Histogram

# Prometheus metrics
INFERENCE_COUNT = Counter('neurosql_inferences_total', 'Total inferences processed')
RISK_SCORE = Gauge('neurosql_risk_score', 'Current risk score')
CONFIDENCE_HISTOGRAM = Histogram('neurosql_confidence', 'Confidence distribution')
EVIDENCE_QUALITY = Gauge('neurosql_evidence_quality', 'Evidence quality score')
BLOCKED_QUERIES = Counter('neurosql_queries_blocked_total', 'Queries blocked for safety')

@dataclass
class RiskMetrics:
    """Track risk metrics over time"""
    timestamp: datetime
    total_inferences: int = 0
    blocked_queries: int = 0
    high_risk_inferences: int = 0
    average_confidence: float = 0.0
    evidence_quality_score: float = 0.0
    system_memory_usage: float = 0.0
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_inferences': self.total_inferences,
            'blocked_queries': self.blocked_queries,
            'high_risk_inferences': self.high_risk_inferences,
            'average_confidence': self.average_confidence,
            'evidence_quality_score': self.evidence_quality_score,
            'system_memory_usage': self.system_memory_usage
        }

class DeploymentMonitor:
    """Monitor deployment health and risks"""
    
    def __init__(self, port: int = 9090):
        self.metrics_history: List[RiskMetrics] = []
        self.risk_thresholds = {
            'high_risk_percentage': 0.1,  # Alert if >10% high risk
            'blocked_query_percentage': 0.05,  # Alert if >5% blocked
            'low_confidence_threshold': 0.3,  # Alert if avg confidence <30%
            'memory_threshold': 0.8  # Alert if memory >80%
        }
        self.alerts = []
        self.logger = logging.getLogger(__name__)
        
        # Start Prometheus server
        start_http_server(port)
        self.logger.info(f"Monitoring started on port {port}")
    
    def record_inference(self, confidence: float, risk_level: str, 
                        blocked: bool = False, evidence_quality: float = 0.5):
        """Record an inference for monitoring"""
        
        # Update Prometheus metrics
        INFERENCE_COUNT.inc()
        CONFIDENCE_HISTOGRAM.observe(confidence)
        EVIDENCE_QUALITY.set(evidence_quality)
        
        if blocked:
            BLOCKED_QUERIES.inc()
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(confidence, risk_level, evidence_quality)
        RISK_SCORE.set(risk_score)
        
        # Check for alerts
        self._check_alerts(confidence, risk_level, blocked)
    
    def _calculate_risk_score(self, confidence: float, risk_level: str, 
                            evidence_quality: float) -> float:
        """Calculate overall risk score"""
        score = 0
        
        # Low confidence increases risk
        if confidence < 0.3:
            score += 0.4
        
        # High confidence with low evidence increases risk
        if confidence > 0.8 and evidence_quality < 0.3:
            score += 0.3
        
        # Risk level contributions
        if risk_level == 'HIGH':
            score += 0.3
        elif risk_level == 'MEDIUM':
            score += 0.15
        
        return min(1.0, score)
    
    def _check_alerts(self, confidence: float, risk_level: str, blocked: bool):
        """Check for alert conditions"""
        
        # Check system resources
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > self.risk_thresholds['memory_threshold'] * 100:
            self._create_alert('HIGH_MEMORY', 
                             f"Memory usage at {memory_percent}%")
        
        # Check risk patterns
        if risk_level == 'HIGH' and confidence > 0.8:
            self._create_alert('HIGH_RISK_HIGH_CONFIDENCE',
                             f"High risk ({risk_level}) with high confidence ({confidence:.1%})")
        
        if blocked:
            self._create_alert('QUERY_BLOCKED',
                             "Query blocked for safety reasons")
    
    def _create_alert(self, alert_type: str, message: str):
        """Create an alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'acknowledged': False
        }
        
        self.alerts.append(alert)
        self.logger.warning(f"ALERT: {alert_type} - {message}")
    
    def generate_health_report(self, hours: int = 24) -> Dict:
        """Generate health report for time period"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history 
                         if m.timestamp > cutoff]
        
        if not recent_metrics:
            return {'message': 'No recent metrics'}
        
        total = len(recent_metrics)
        blocked = sum(1 for m in recent_metrics if m.blocked_queries > 0)
        high_risk = sum(1 for m in recent_metrics if m.high_risk_inferences > 0)
        
        return {
            'period_hours': hours,
            'total_inferences': sum(m.total_inferences for m in recent_metrics),
            'blocked_queries': blocked,
            'blocked_percentage': blocked / total if total > 0 else 0,
            'high_risk_inferences': high_risk,
            'high_risk_percentage': high_risk / total if total > 0 else 0,
            'average_confidence': sum(m.average_confidence for m in recent_metrics) / total if total > 0 else 0,
            'average_evidence_quality': sum(m.evidence_quality_score for m in recent_metrics) / total if total > 0 else 0,
            'system_health': self._calculate_system_health(recent_metrics),
            'active_alerts': len([a for a in self.alerts if not a['acknowledged']]),
            'recommendations': self._generate_recommendations(recent_metrics)
        }
    
    def _calculate_system_health(self, metrics: List[RiskMetrics]) -> str:
        """Calculate overall system health"""
        
        if not metrics:
            return 'UNKNOWN'
        
        recent = metrics[-1]
        
        # Check thresholds
        issues = []
        
        if recent.high_risk_inferences > 0:
            issues.append('high_risk_present')
        
        if recent.average_confidence < self.risk_thresholds['low_confidence_threshold']:
            issues.append('low_confidence')
        
        if recent.system_memory_usage > self.risk_thresholds['memory_threshold']:
            issues.append('high_memory')
        
        if not issues:
            return 'HEALTHY'
        elif len(issues) == 1:
            return 'WARNING'
        else:
            return 'CRITICAL'
    
    def _generate_recommendations(self, metrics: List[RiskMetrics]) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        if not metrics:
            return recommendations
        
        recent = metrics[-1]
        
        # High blocked query rate
        blocked_rate = recent.blocked_queries / max(recent.total_inferences, 1)
        if blocked_rate > self.risk_thresholds['blocked_query_percentage']:
            recommendations.append(
                f"High blocked query rate ({blocked_rate:.1%}). "
                "Consider reviewing clinical safety rules."
            )
        
        # Low evidence quality
        if recent.evidence_quality_score < 0.5:
            recommendations.append(
                f"Low evidence quality ({recent.evidence_quality_score:.1%}). "
                "Consider improving evidence validation."
            )
        
        return recommendations
    
    def run_monitoring_dashboard(self):
        """Run monitoring dashboard (simplified)"""
        print("NeuroSQL Deployment Monitor")
        print("=" * 60)
        
        while True:
            # Collect system metrics
            memory = psutil.virtual_memory().percent / 100
            cpu = psutil.cpu_percent() / 100
            
            # Create metrics snapshot
            metrics = RiskMetrics(
                timestamp=datetime.now(),
                system_memory_usage=memory,
                # Other metrics would come from actual inference tracking
            )
            
            self.metrics_history.append(metrics)
            
            # Keep history manageable
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            # Print status
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
            print(f"Memory: {memory:.1%} | CPU: {cpu:.1%}")
            print(f"Alerts: {len([a for a in self.alerts if not a['acknowledged']])} active")
            
            # Generate report every 5 minutes
            if len(self.metrics_history) % 5 == 0:
                report = self.generate_health_report(1)  # Last hour
                if 'system_health' in report:
                    print(f"System Health: {report['system_health']}")
            
            time.sleep(60)  # Update every minute

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroSQL Deployment Monitor")
    parser.add_argument("--port", type=int, default=9090, help="Prometheus port")
    parser.add_argument("--dashboard", action="store_true", help="Run dashboard")
    
    args = parser.parse_args()
    
    monitor = DeploymentMonitor(args.port)
    
    if args.dashboard:
        monitor.run_monitoring_dashboard()
    else:
        print(f"Monitoring running on port {args.port}")
        print("Metrics available at http://localhost:{args.port}/metrics")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Monitoring stopped")
