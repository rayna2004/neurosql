# ============================================================================
# SAFETY BOUNDARIES & REGULATORY CONSTRAINTS
# ============================================================================

import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
import hashlib
import sqlite3
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# REGULATORY COMPLIANCE FRAMEWORK
# ============================================================================

class RegulatoryCompliance:
    """Regulatory compliance framework for clinical AI systems"""
    
    def __init__(self):
        self.regulations = self._load_regulations()
        self.compliance_log = []
        self.audit_trail = []
        
    def _load_regulations(self) -> Dict:
        """Load regulatory requirements"""
        return {
            'hipaa': {
                'name': 'HIPAA Privacy Rule',
                'requirements': [
                    'no_patient_identification',
                    'data_encryption_at_rest',
                    'access_controls',
                    'audit_trails'
                ],
                'applicable': True,
                'enforcement_level': 'legal'
            },
            'fda_ai': {
                'name': 'FDA AI/ML Software as Medical Device',
                'requirements': [
                    'clinical_validation',
                    'explainability',
                    'performance_monitoring',
                    'change_control'
                ],
                'applicable': True,
                'enforcement_level': 'regulatory'
            },
            'gdpr': {
                'name': 'GDPR Data Protection',
                'requirements': [
                    'data_minimization',
                    'purpose_limitation',
                    'right_to_explanation'
                ],
                'applicable': True,
                'enforcement_level': 'legal'
            },
            'ich_gcp': {
                'name': 'ICH Good Clinical Practice',
                'requirements': [
                    'protocol_compliance',
                    'informed_consent',
                    'safety_monitoring'
                ],
                'applicable': True,
                'enforcement_level': 'regulatory'
            }
        }
    
    def check_compliance(self, system_state: Dict, clinical_mode: bool = False) -> Dict:
        """Check system compliance with regulations"""
        
        compliance_report = {
            'timestamp': datetime.now().isoformat(),
            'clinical_mode': clinical_mode,
            'checks_performed': [],
            'violations': [],
            'warnings': [],
            'overall_compliance': True
        }
        
        # Check each regulation
        for reg_id, reg_info in self.regulations.items():
            if clinical_mode or reg_info['applicable']:
                check_result = self._check_single_regulation(reg_id, reg_info, system_state)
                compliance_report['checks_performed'].append(check_result)
                
                if check_result['violations']:
                    compliance_report['violations'].extend(check_result['violations'])
                    compliance_report['overall_compliance'] = False
                
                if check_result['warnings']:
                    compliance_report['warnings'].extend(check_result['warnings'])
        
        # Log compliance check
        self.compliance_log.append(compliance_report)
        
        # Maintain audit trail
        self._add_to_audit_trail('compliance_check', compliance_report)
        
        return compliance_report
    
    def _check_single_regulation(self, reg_id: str, reg_info: Dict, system_state: Dict) -> Dict:
        """Check compliance with a single regulation"""
        
        violations = []
        warnings = []
        
        # Check each requirement
        for requirement in reg_info['requirements']:
            check_method = getattr(self, f'_check_{requirement}', None)
            if check_method:
                result = check_method(system_state)
                if not result['compliant']:
                    if reg_info['enforcement_level'] == 'legal':
                        violations.append({
                            'regulation': reg_info['name'],
                            'requirement': requirement,
                            'issue': result['issue'],
                            'severity': 'critical'
                        })
                    else:
                        warnings.append({
                            'regulation': reg_info['name'],
                            'requirement': requirement,
                            'issue': result['issue'],
                            'severity': 'warning'
                        })
        
        return {
            'regulation': reg_info['name'],
            'regulation_id': reg_id,
            'enforcement_level': reg_info['enforcement_level'],
            'violations': violations,
            'warnings': warnings,
            'compliant': len(violations) == 0
        }
    
    def _check_no_patient_identification(self, system_state: Dict) -> Dict:
        """Check for patient identification data"""
        # In production, would scan all data for PHI
        test_phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{5}-\d{4}\b',  # ZIP+4
            r'\b\d{10}\b',  # Phone
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'  # Full name pattern
        ]
        
        data_sample = json.dumps(system_state).lower()
        
        for pattern in test_phi_patterns:
            if re.search(pattern, data_sample):
                return {
                    'compliant': False,
                    'issue': f'Potential PHI detected with pattern: {pattern}'
                }
        
        return {'compliant': True, 'issue': None}
    
    def _check_data_encryption_at_rest(self, system_state: Dict) -> Dict:
        """Check data encryption"""
        # Simplified check
        if system_state.get('encryption_enabled', False):
            return {'compliant': True, 'issue': None}
        else:
            return {
                'compliant': False,
                'issue': 'Data encryption at rest not confirmed'
            }
    
    def _check_access_controls(self, system_state: Dict) -> Dict:
        """Check access controls"""
        if system_state.get('access_controls', {}).get('enabled', False):
            return {'compliant': True, 'issue': None}
        else:
            return {
                'compliant': False,
                'issue': 'Access controls not properly configured'
            }
    
    def _check_audit_trails(self, system_state: Dict) -> Dict:
        """Check audit trail implementation"""
        if system_state.get('audit_trail', {}).get('enabled', False):
            return {'compliant': True, 'issue': None}
        else:
            return {
                'compliant': False,
                'issue': 'Audit trail not properly implemented'
            }
    
    def _check_clinical_validation(self, system_state: Dict) -> Dict:
        """Check clinical validation"""
        if system_state.get('validation', {}).get('clinical_validation_performed', False):
            return {'compliant': True, 'issue': None}
        else:
            return {
                'compliant': False,
                'issue': 'Clinical validation not performed'
            }
    
    def _check_explainability(self, system_state: Dict) -> Dict:
        """Check explainability requirements"""
        if system_state.get('explainability', {}).get('enabled', False):
            return {'compliant': True, 'issue': None}
        else:
            return {
                'compliant': False,
                'issue': 'Explainability features not enabled'
            }
    
    def _add_to_audit_trail(self, action: str, data: Dict):
        """Add entry to audit trail"""
        audit_entry = {
            'audit_id': "audit_{hashlib.md5(f"{datetime.now().isoformat()}_{action}".encode()).hexdigest()[:16]}",
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'data': data,
            'user': 'system',
            'ip_address': '127.0.0.1'
        }
        
        self.audit_trail.append(audit_entry)
        
        # Keep audit trail manageable
        if len(self.audit_trail) > 10000:
            self.audit_trail = self.audit_trail[-5000:]
    
    def generate_compliance_report(self, days: int = 30) -> Dict:
        """Generate compliance report"""
        cutoff = datetime.now() - timedelta(days=days)
        
        recent_checks = [
            check for check in self.compliance_log 
            if datetime.fromisoformat(check['timestamp']) > cutoff
        ]
        
        if not recent_checks:
            return {'message': 'No compliance checks in specified period'}
        
        total_checks = len(recent_checks)
        compliant_checks = sum(1 for check in recent_checks if check['overall_compliance'])
        violation_count = sum(len(check['violations']) for check in recent_checks)
        warning_count = sum(len(check['warnings']) for check in recent_checks)
        
        # Most common violations
        all_violations = []
        for check in recent_checks:
            all_violations.extend(check['violations'])
        
        violation_types = {}
        for violation in all_violations:
            key = f"{violation['regulation']}:{violation['requirement']}"
            violation_types[key] = violation_types.get(key, 0) + 1
        
        common_violations = sorted(
            [{'type': k, 'count': v} for k, v in violation_types.items()],
            key=lambda x: x['count'],
            reverse=True
        )[:5]
        
        return {
            'report_period_days': days,
            'total_compliance_checks': total_checks,
            'compliant_checks': compliant_checks,
            'compliance_rate': compliant_checks / total_checks if total_checks > 0 else 0,
            'total_violations': violation_count,
            'total_warnings': warning_count,
            'most_common_violations': common_violations,
            'compliance_status': 'COMPLIANT' if compliant_checks == total_checks else 'NON_COMPLIANT',
            'recommendations': self._generate_compliance_recommendations(recent_checks)
        }
    
    def _generate_compliance_recommendations(self, recent_checks: List[Dict]) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        # Check for persistent violations
        all_violations = []
        for check in recent_checks:
            all_violations.extend([(v['regulation'], v['requirement']) for v in check['violations']])
        
        from collections import Counter
        violation_counts = Counter(all_violations)
        
        for (regulation, requirement), count in violation_counts.most_common(3):
            if count >= 3:  # Persistent violation
                recommendations.append(
                    f"Address persistent violation: {regulation} - {requirement} "
                    f"(occurred {count} times)"
                )
        
        # Check audit trail completeness
        if len(self.audit_trail) < 100:
            recommendations.append(
                "Audit trail has limited entries. Ensure all critical actions are logged."
            )
        
        if not recommendations:
            recommendations.append("Compliance status is good. Continue regular monitoring.")
        
        return recommendations

# ============================================================================
# DIAGNOSTIC GUARDRAILS
# ============================================================================

class DiagnosticGuardrails:
    """Guardrails to prevent diagnostic misuse"""
    
    def __init__(self):
        self.blocked_queries = set()
        self.warning_queries = {}
        self.diagnostic_log = []
        
        # FDA-prohibited diagnostic claims
        self.prohibited_diagnostic_terms = [
            'diagnose', 'diagnosis', 'diagnostic',
            'treat', 'treatment', 'therapy', 'therapeutic',
            'cure', 'heal', 'remedy',
            'prescribe', 'prescription', 'medication',
            'dose', 'dosage', 'mg', 'milligram',
            'should take', 'must take', 'recommend taking'
        ]
        
        # High-risk disease terms
        self.high_risk_diseases = [
            'cancer', 'tumor', 'malignant',
            'alzheimer', 'dementia',
            'schizophrenia', 'psychosis',
            'stroke', 'heart attack', 'myocardial infarction',
            'sepsis', 'meningitis'
        ]
    
    def check_query_safety(self, query: str, user_context: Dict = None) -> Dict:
        """Check if query is safe for diagnostic context"""
        
        query_lower = query.lower()
        
        # Check if query is blocked
        if query in self.blocked_queries:
            return {
                'safe': False,
                'blocked': True,
                'reason': 'Previously blocked query',
                'severity': 'high'
            }
        
        # Check for prohibited diagnostic terms
        prohibited_found = []
        for term in self.prohibited_diagnostic_terms:
            if term in query_lower:
                prohibited_found.append(term)
        
        if prohibited_found:
            return {
                'safe': False,
                'blocked': True,
                'reason': f'Contains prohibited diagnostic terms: {", ".join(prohibited_found)}',
                'severity': 'high',
                'terms_found': prohibited_found
            }
        
        # Check for high-risk disease terms
        high_risk_found = []
        for disease in self.high_risk_diseases:
            if disease in query_lower:
                high_risk_found.append(disease)
        
        if high_risk_found:
            warning = {
                'safe': True,
                'blocked': False,
                'warning': True,
                'reason': f'Contains high-risk disease terms: {", ".join(high_risk_found)}',
                'severity': 'medium',
                'terms_found': high_risk_found,
                'recommendation': 'Consult healthcare professional for medical advice'
            }
            
            # Add to warning log
            warning_key = hashlib.md5(query.encode()).hexdigest()
            self.warning_queries[warning_key] = warning
            
            return warning
        
        # Check user context if provided
        if user_context:
            if not user_context.get('medical_training', False):
                # Non-medical users get extra warnings
                medical_terms = ['symptom', 'symptoms', 'sign', 'signs', 'test', 'tests']
                if any(term in query_lower for term in medical_terms):
                    return {
                        'safe': True,
                        'blocked': False,
                        'warning': True,
                        'reason': 'Non-medical user querying medical symptoms',
                        'severity': 'low',
                        'recommendation': 'This information is for educational purposes only'
                    }
        
        return {
            'safe': True,
            'blocked': False,
            'warning': False,
            'reason': 'Query passed all safety checks'
        }
    
    def apply_diagnostic_safeguards(self, result: Dict, query: str) -> Dict:
        """Apply diagnostic safeguards to results"""
        
        safety_check = self.check_query_safety(query)
        
        if not safety_check['safe']:
            return {
                'error': 'Query blocked by diagnostic guardrails',
                'safety_check': safety_check,
                'original_result': None
            }
        
        # Add mandatory disclaimers
        if 'disclaimers' not in result:
            result['disclaimers'] = []
        
        disclaimer = """
        ⚕️ MEDICAL DISCLAIMER:
        This information is for educational and research purposes only.
        It is not medical advice, diagnosis, or treatment.
        Always consult qualified healthcare professionals for medical concerns.
        Do not delay seeking medical attention based on this information.
        """
        
        result['disclaimers'].append({
            'type': 'medical_safety',
            'text': disclaimer,
            'timestamp': datetime.now().isoformat(),
            'required': True
        })
        
        # Cap confidence for diagnostic-like outputs
        if result.get('confidence', 0) > 0.8:
            result['original_confidence'] = result['confidence']
            result['confidence'] = 0.8  # Conservative cap
            result['confidence_capped'] = True
            result['capping_reason'] = 'Diagnostic safety precaution'
        
        # Add warning if safety check flagged
        if safety_check.get('warning', False):
            result['safety_warnings'] = result.get('safety_warnings', [])
            result['safety_warnings'].append({
                'type': 'diagnostic_safety',
                'message': safety_check['reason'],
                'severity': safety_check['severity'],
                'recommendation': safety_check.get('recommendation', '')
            })
        
        # Log diagnostic access
        self._log_diagnostic_access(query, result, safety_check)
        
        return result
    
    def _log_diagnostic_access(self, query: str, result: Dict, safety_check: Dict):
        """Log diagnostic query access"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'query_hash': hashlib.md5(query.encode()).hexdigest(),
            'safety_check': safety_check,
            'result_confidence': result.get('confidence', 0),
            'disclaimers_added': len(result.get('disclaimers', [])),
            'warnings_added': len(result.get('safety_warnings', [])),
            'confidence_capped': result.get('confidence_capped', False)
        }
        
        self.diagnostic_log.append(log_entry)
    
    def get_diagnostic_safety_report(self) -> Dict:
        """Get diagnostic safety report"""
        if not self.diagnostic_log:
            return {'message': 'No diagnostic queries processed'}
        
        total_queries = len(self.diagnostic_log)
        blocked_queries = sum(1 for entry in self.diagnostic_log 
                             if entry['safety_check'].get('blocked', False))
        warned_queries = sum(1 for entry in self.diagnostic_log 
                            if entry['safety_check'].get('warning', False))
        capped_confidence = sum(1 for entry in self.diagnostic_log 
                               if entry.get('confidence_capped', False))
        
        # Analyze query patterns
        query_patterns = {}
        for entry in self.diagnostic_log[-100:]:  # Last 100 queries
            query_lower = entry['query'].lower()
            
            # Check for common patterns
            if any(term in query_lower for term in self.prohibited_diagnostic_terms):
                pattern = 'prohibited_diagnostic_terms'
            elif any(disease in query_lower for disease in self.high_risk_diseases):
                pattern = 'high_risk_disease'
            else:
                pattern = 'general_neuroscience'
            
            query_patterns[pattern] = query_patterns.get(pattern, 0) + 1
        
        return {
            'total_diagnostic_queries': total_queries,
            'blocked_queries': blocked_queries,
            'blocked_percentage': blocked_queries / total_queries if total_queries > 0 else 0,
            'warned_queries': warned_queries,
            'confidence_capped_queries': capped_confidence,
            'query_patterns': query_patterns,
            'safety_status': 'SAFE' if blocked_queries == 0 else 'NEEDS_REVIEW',
            'recommendations': self._generate_safety_recommendations(blocked_queries, warned_queries)
        }
    
    def _generate_safety_recommendations(self, blocked: int, warned: int) -> List[str]:
        """Generate safety recommendations"""
        recommendations = []
        
        if blocked > 0:
            recommendations.append(
                f"{blocked} queries were blocked. Review blocked queries for false positives."
            )
        
        if warned > 0:
            recommendations.append(
                f"{warned} queries triggered warnings. Consider user education about appropriate use."
            )
        
        if blocked == 0 and warned == 0:
            recommendations.append(
                "Diagnostic safety guardrails are working correctly. Continue monitoring."
            )
        
        return recommendations

# ============================================================================
# HUMAN-IN-THE-LOOP ENFORCEMENT
# ============================================================================

class HumanInTheLoop:
    """Human-in-the-loop enforcement system"""
    
    def __init__(self, db_path: str = "data/human_review.db"):
        self.db_path = db_path
        self._init_database()
        self.reviewers = {}
        
    def _init_database(self):
        """Initialize human review database"""
        Path("data").mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Review queue
            conn.execute("""
                CREATE TABLE IF NOT EXISTS review_queue (
                    review_id TEXT PRIMARY KEY,
                    inference_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    result_data TEXT NOT NULL,
                    confidence REAL,
                    uncertainty TEXT,
                    risk_level TEXT,
                    priority INTEGER DEFAULT 5,
                    status TEXT DEFAULT 'pending',
                    assigned_to TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    assigned_at TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            
            # Review decisions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS review_decisions (
                    decision_id TEXT PRIMARY KEY,
                    review_id TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    confidence_correction REAL,
                    notes TEXT,
                    decision_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (review_id) REFERENCES review_queue(review_id)
                )
            """)
            
            # Reviewer performance
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reviewer_performance (
                    reviewer_id TEXT PRIMARY KEY,
                    total_reviews INTEGER DEFAULT 0,
                    average_review_time_seconds REAL DEFAULT 0,
                    agreement_rate REAL DEFAULT 0,
                    last_active TIMESTAMP,
                    qualifications TEXT
                )
            """)
            
            # Escalation rules
            conn.execute("""
                CREATE TABLE IF NOT EXISTS escalation_rules (
                    rule_id TEXT PRIMARY KEY,
                    condition_type TEXT NOT NULL,
                    condition_value REAL,
                    action TEXT NOT NULL,
                    priority INTEGER DEFAULT 1,
                    enabled BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_status ON review_queue(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_priority ON review_queue(priority)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_review ON review_decisions(review_id)")
            
            conn.commit()
            
            # Initialize default escalation rules
            self._initialize_escalation_rules(conn)
    
    def _initialize_escalation_rules(self, conn: sqlite3.Connection):
        """Initialize default escalation rules"""
        default_rules = [
            ('high_risk_clinical', 'risk_level', 'HIGH', 'require_senior_review', 1, True),
            ('low_confidence_high_impact', 'confidence', 0.3, 'flag_for_review', 2, True),
            ('contradictory_evidence', 'uncertainty', 0.7, 'escalate_to_expert', 1, True),
            ('novel_finding', 'confidence', 0.8, 'flag_for_validation', 3, True)
        ]
        
        for rule_id, condition_type, condition_value, action, priority, enabled in default_rules:
            conn.execute("""
                INSERT OR IGNORE INTO escalation_rules 
                (rule_id, condition_type, condition_value, action, priority, enabled)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (rule_id, condition_type, condition_value, action, priority, enabled))
        
        conn.commit()
    
    def check_requires_human_review(self, inference_result: Dict) -> bool:
        """Check if inference requires human review"""
        
        # Check escalation rules
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT rule_id, action FROM escalation_rules 
                WHERE enabled = TRUE 
                ORDER BY priority
            """)
            
            rules = cursor.fetchall()
            
            for rule_id, action in rules:
                if self._check_rule_condition(rule_id, inference_result):
                    logger.info(f"Rule {rule_id} triggered: {action}")
                    return True
        
        return False
    
    def _check_rule_condition(self, rule_id: str, inference_result: Dict) -> bool:
        """Check if a rule condition is met"""
        
        if rule_id == 'high_risk_clinical':
            return inference_result.get('risk_level') == 'HIGH'
        
        elif rule_id == 'low_confidence_high_impact':
            confidence = inference_result.get('confidence', 1)
            # Check if it's high impact (simplified)
            high_impact_terms = ['treatment', 'diagnosis', 'clinical', 'patient']
            query = inference_result.get('query', '').lower()
            is_high_impact = any(term in query for term in high_impact_terms)
            return confidence < 0.3 and is_high_impact
        
        elif rule_id == 'contradictory_evidence':
            uncertainty = inference_result.get('uncertainty', {}).get('level', 'LOW')
            return uncertainty == 'HIGH'
        
        elif rule_id == 'novel_finding':
            confidence = inference_result.get('confidence', 0)
            evidence_count = inference_result.get('evidence_count', 0)
            # High confidence with low evidence suggests novel finding
            return confidence > 0.8 and evidence_count < 2
        
        return False
    
    def add_to_review_queue(self, inference_id: str, query: str, 
                           result_data: Dict, priority: int = 5):
        """Add inference to human review queue"""
        
        review_id = "rev_{hashlib.md5(f"{inference_id}_{datetime.now()}".encode()).hexdigest()[:16]}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO review_queue 
                (review_id, inference_id, query, result_data, confidence, 
                 uncertainty, risk_level, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                review_id,
                inference_id,
                query,
                json.dumps(result_data),
                result_data.get('confidence', 0),
                json.dumps(result_data.get('uncertainty', {})),
                result_data.get('risk_level', 'MEDIUM'),
                priority
            ))
            conn.commit()
        
        logger.info(f"Added inference {inference_id} to review queue as {review_id}")
        return review_id
    
    def get_next_review_item(self, reviewer_id: str) -> Optional[Dict]:
        """Get next item for human review"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Find highest priority pending review
            cursor = conn.execute("""
                SELECT * FROM review_queue 
                WHERE status = 'pending' 
                ORDER BY priority DESC, created_at ASC 
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Assign to reviewer
            review_id = row['review_id']
            conn.execute("""
                UPDATE review_queue 
                SET status = 'assigned', assigned_to = ?, assigned_at = CURRENT_TIMESTAMP
                WHERE review_id = ?
            """, (reviewer_id, review_id))
            conn.commit()
            
            # Prepare review item
            review_item = dict(row)
            review_item['result_data'] = json.loads(review_item['result_data'])
            if review_item['uncertainty']:
                review_item['uncertainty'] = json.loads(review_item['uncertainty'])
            
            return review_item
    
    def submit_review_decision(self, review_id: str, reviewer_id: str,
                              decision: str, confidence_correction: float = None,
                              notes: str = ""):
        """Submit human review decision"""
        
        decision_id = "dec_{hashlib.md5(f"{review_id}_{decision}".encode()).hexdigest()[:16]}"
        
        with sqlite3.connect(self.db_path) as conn:
            # Record decision
            conn.execute("""
                INSERT INTO review_decisions 
                (decision_id, review_id, reviewer_id, decision, confidence_correction, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (decision_id, review_id, reviewer_id, decision, confidence_correction, notes))
            
            # Update review status
            conn.execute("""
                UPDATE review_queue 
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE review_id = ?
            """, (review_id,))
            
            # Update reviewer performance
            self._update_reviewer_performance(reviewer_id, conn)
            
            conn.commit()
        
        logger.info(f"Review {review_id} completed by {reviewer_id}: {decision}")
        
        return decision_id
    
    def _update_reviewer_performance(self, reviewer_id: str, conn: sqlite3.Connection):
        """Update reviewer performance metrics"""
        # Get reviewer stats
        cursor = conn.execute("""
            SELECT COUNT(*) as total, 
                   AVG(JULIANDAY(decision_time) - JULIANDAY(assigned_at)) * 86400 as avg_time
            FROM review_decisions d
            JOIN review_queue q ON d.review_id = q.review_id
            WHERE d.reviewer_id = ?
        """, (reviewer_id,))
        
        stats = cursor.fetchone()
        
        if stats and stats[0] > 0:
            conn.execute("""
                INSERT OR REPLACE INTO reviewer_performance 
                (reviewer_id, total_reviews, average_review_time_seconds, last_active)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (reviewer_id, stats[0], stats[1]))
    
    def get_review_queue_status(self) -> Dict:
        """Get review queue status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count 
                FROM review_queue 
                GROUP BY status
            """)
            status_counts = {row[0]: row[1] for row in cursor}
            
            cursor = conn.execute("""
                SELECT AVG(JULIANDAY('now') - JULIANDAY(created_at)) * 24 as avg_wait_hours
                FROM review_queue 
                WHERE status = 'pending'
            """)
            avg_wait = cursor.fetchone()[0] or 0
            
            cursor = conn.execute("SELECT COUNT(*) FROM review_decisions")
            total_decisions = cursor.fetchone()[0]
        
        return {
            'queue_status': status_counts,
            'pending_count': status_counts.get('pending', 0),
            'in_progress_count': status_counts.get('assigned', 0),
            'completed_count': status_counts.get('completed', 0),
            'average_wait_hours': avg_wait,
            'total_decisions_made': total_decisions,
            'human_review_enabled': True,
            'queue_health': 'HEALTHY' if avg_wait < 24 else 'BACKLOG'
        }

# ============================================================================
# SAFETY MONITORING & ALERTING
# ============================================================================

class SafetyMonitor:
    """Monitor safety violations and trigger alerts"""
    
    def __init__(self):
        self.safety_events = []
        self.alerts = []
        self.alert_thresholds = {
            'blocked_queries_per_hour': 10,
            'high_risk_inferences_per_day': 5,
            'compliance_violations_per_week': 3,
            'human_review_backlog_hours': 24
        }
    
    def monitor_safety_event(self, event_type: str, event_data: Dict):
        """Monitor a safety event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'event_data': event_data,
            'severity': self._determine_severity(event_type, event_data)
        }
        
        self.safety_events.append(event)
        
        # Check for alerts
        self._check_alerts(event)
        
        # Log event
        logger.info(f"Safety event: {event_type} - {event['severity']}")
    
    def _determine_severity(self, event_type: str, event_data: Dict) -> str:
        """Determine event severity"""
        if event_type == 'query_blocked':
            return 'HIGH'
        elif event_type == 'compliance_violation':
            return event_data.get('severity', 'MEDIUM')
        elif event_type == 'human_review_required':
            return 'MEDIUM'
        elif event_type == 'confidence_capped':
            return 'LOW'
        else:
            return 'INFO'
    
    def _check_alerts(self, event: Dict):
        """Check if event triggers an alert"""
        
        # Check blocked queries rate
        if event['event_type'] == 'query_blocked':
            recent_blocked = len([
                e for e in self.safety_events[-100:]
                if e['event_type'] == 'query_blocked' 
                and datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(hours=1)
            ])
            
            if recent_blocked >= self.alert_thresholds['blocked_queries_per_hour']:
                self._trigger_alert(
                    'HIGH_BLOCKED_QUERY_RATE',
                    f"{recent_blocked} queries blocked in the last hour",
                    {'blocked_count': recent_blocked, 'threshold': self.alert_thresholds['blocked_queries_per_hour']}
                )
        
        # Check high risk inferences
        if event['severity'] == 'HIGH':
            recent_high_risk = len([
                e for e in self.safety_events[-500:]
                if e['severity'] == 'HIGH'
                and datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(days=1)
            ])
            
            if recent_high_risk >= self.alert_thresholds['high_risk_inferences_per_day']:
                self._trigger_alert(
                    'HIGH_RISK_INFERENCE_RATE',
                    f"{recent_high_risk} high-risk inferences in the last day",
                    {'high_risk_count': recent_high_risk, 'threshold': self.alert_thresholds['high_risk_inferences_per_day']}
                )
    
    def _trigger_alert(self, alert_type: str, message: str, details: Dict):
        """Trigger an alert"""
        alert = {
            'alert_id': "alert_{hashlib.md5(f"{alert_type}_{datetime.now()}".encode()).hexdigest()[:16]}",
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'details': details,
            'acknowledged': False,
            'escalated': False
        }
        
        self.alerts.append(alert)
        
        # Log alert
        logger.warning(f"SAFETY ALERT: {alert_type} - {message}")
        
        # In production, would trigger notifications (email, Slack, etc.)
    
    def get_safety_report(self) -> Dict:
        """Get safety monitoring report"""
        
        # Event statistics
        event_types = {}
        severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'INFO': 0}
        
        for event in self.safety_events[-1000:]:  # Last 1000 events
            event_types[event['event_type']] = event_types.get(event['event_type'], 0) + 1
            severity_counts[event['severity']] = severity_counts.get(event['severity'], 0) + 1
        
        # Active alerts
        active_alerts = [a for a in self.alerts if not a['acknowledged']]
        
        # Rate calculations
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        recent_events = [e for e in self.safety_events if datetime.fromisoformat(e['timestamp']) > hour_ago]
        events_per_hour = len(recent_events)
        
        high_risk_recent = [e for e in recent_events if e['severity'] == 'HIGH']
        high_risk_per_hour = len(high_risk_recent)
        
        return {
            'total_events_monitored': len(self.safety_events),
            'events_last_hour': events_per_hour,
            'high_risk_events_last_hour': high_risk_per_hour,
            'event_type_distribution': event_types,
            'severity_distribution': severity_counts,
            'active_alerts': len(active_alerts),
            'recent_alerts': active_alerts[:5],
            'safety_status': self._determine_safety_status(high_risk_per_hour),
            'recommendations': self._generate_safety_recommendations(high_risk_per_hour, active_alerts)
        }
    
    def _determine_safety_status(self, high_risk_per_hour: int) -> str:
        """Determine overall safety status"""
        if high_risk_per_hour > 10:
            return 'CRITICAL'
        elif high_risk_per_hour > 5:
            return 'WARNING'
        elif high_risk_per_hour > 1:
            return 'CAUTION'
        else:
            return 'NORMAL'
    
    def _generate_safety_recommendations(self, high_risk_per_hour: int, 
                                       active_alerts: List[Dict]) -> List[str]:
        """Generate safety recommendations"""
        recommendations = []
        
        if high_risk_per_hour > 5:
            recommendations.append(
                f"High rate of high-risk events ({high_risk_per_hour}/hour). "
                "Consider tightening safety thresholds."
            )
        
        if len(active_alerts) > 3:
            recommendations.append(
                f"{len(active_alerts)} active alerts. Review and acknowledge alerts."
            )
        
        if not recommendations:
            recommendations.append("Safety monitoring indicates normal operation.")
        
        return recommendations
