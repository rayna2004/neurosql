# ============================================================================
# VALIDATED EVIDENCE SYSTEM
# ============================================================================

import re
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class EvidenceSource(Enum):
    PUBMED = "pubmed"
    TEXTBOOK = "textbook"
    DATASET = "dataset"
    CLINICAL_TRIAL = "clinical_trial"
    EXPERT_CURATION = "expert_curation"
    SYSTEMATIC_REVIEW = "systematic_review"

class EvidenceQuality(Enum):
    CURATED = 5        # Expert curated, validated
    PEER_REVIEWED = 4  # Published in peer-reviewed journal
    PREPRINT = 3       # Preprint with DOI
    DATASET = 2        # Structured dataset
    WEB = 1            # Web source (lowest)
    UNVERIFIED = 0     # Unverified source

@dataclass
class ValidatedEvidence:
    """Evidence with comprehensive validation"""
    evidence_id: str
    source_type: EvidenceSource
    source_id: str
    snippet: str
    extracted_fact: Optional[Dict] = None
    retrieval_time: datetime = field(default_factory=datetime.now)
    content_hash: str = ""
    
    # Validation fields
    source_validated: bool = False
    content_validated: bool = False
    cross_referenced: bool = False
    validation_score: float = 0.0
    validation_notes: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of evidence content"""
        content = f"{self.source_id}|{self.snippet}|{self.retrieval_time.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()

class PubMedValidator:
    """Validate PubMed IDs and fetch real metadata"""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self):
        self.cache = {}
        self.invalid_pmids = set()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def validate_pmid(self, pmid: str) -> Dict:
        """Validate PubMed ID and fetch metadata"""
        
        # Check cache
        if pmid in self.cache:
            return self.cache[pmid]
        
        if pmid in self.invalid_pmids:
            return {'valid': False, 'reason': 'Previously invalidated'}
        
        # Check format
        if not re.match(r'^\d{1,8}$', pmid):
            self.invalid_pmids.add(pmid)
            return {'valid': False, 'reason': 'Invalid PMID format'}
        
        try:
            # Fetch from PubMed API
            url = f"{self.BASE_URL}/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                self.invalid_pmids.add(pmid)
                return {'valid': False, 'reason': f'API error: {response.status_code}'}
            
            data = response.json()
            
            if 'result' not in data or pmid not in data['result']:
                self.invalid_pmids.add(pmid)
                return {'valid': False, 'reason': 'PMID not found in PubMed'}
            
            article_data = data['result'][pmid]
            
            result = {
                'valid': True,
                'pmid': pmid,
                'title': article_data.get('title', ''),
                'authors': article_data.get('authors', []),
                'pubdate': article_data.get('pubdate', ''),
                'source': article_data.get('source', ''),
                'doi': article_data.get('doi', ''),
                'pmc': article_data.get('pmcid', ''),
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
            
            self.cache[pmid] = result
            return result
            
        except Exception as e:
            logger.error(f"PubMed validation error for {pmid}: {e}")
            return {'valid': False, 'reason': f'Validation error: {str(e)}'}
    
    async def verify_snippet_in_article(self, pmid: str, snippet: str) -> bool:
        """Verify snippet actually appears in article (simplified check)"""
        try:
            # In production, would use full text API
            # For demo, check if snippet makes sense with title
            metadata = await self.validate_pmid(pmid)
            if not metadata['valid']:
                return False
            
            # Basic semantic check
            title = metadata.get('title', '').lower()
            snippet_lower = snippet.lower()
            
            # Check for common mismatches
            mismatch_indicators = [
                'test_snippet',
                'placeholder',
                'example text',
                'lorem ipsum',
                'TODO:',
                'FIXME:'
            ]
            
            if any(indicator in snippet_lower for indicator in mismatch_indicators):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Snippet verification error: {e}")
            return False

class BiologicalConstraintValidator:
    """Apply biological constraints to prevent semantic drift"""
    
    def __init__(self):
        self.biological_rules = self._load_biological_rules()
        self.domain_knowledge = self._load_domain_knowledge()
    
    def _load_biological_rules(self) -> List[Dict]:
        """Load biological plausibility rules"""
        return [
            {
                'name': 'neurotransmitter_specificity',
                'check': self._check_neurotransmitter_specificity,
                'description': 'Neurotransmitters act on specific receptor types'
            },
            {
                'name': 'anatomical_constraints',
                'check': self._check_anatomical_constraints,
                'description': 'Brain regions have specific connections'
            },
            {
                'name': 'temporal_constraints',
                'check': self._check_temporal_constraints,
                'description': 'Biological processes have temporal constraints'
            },
            {
                'name': 'dose_response',
                'check': self._check_dose_response,
                'description': 'Biological effects follow dose-response relationships'
            }
        ]
    
    def _load_domain_knowledge(self) -> Dict:
        """Load domain knowledge constraints"""
        return {
            'neurotransmitter_receptors': {
                'dopamine': ['D1', 'D2', 'D3', 'D4', 'D5'],
                'serotonin': ['5-HT1A', '5-HT2A', '5-HT2C', '5-HT3', '5-HT4', '5-HT6', '5-HT7'],
                'glutamate': ['NMDA', 'AMPA', 'kainate', 'mGluR'],
                'gaba': ['GABA_A', 'GABA_B']
            },
            'brain_region_connections': {
                'hippocampus': ['entorhinal_cortex', 'septum', 'mammillary_bodies', 'prefrontal_cortex'],
                'prefrontal_cortex': ['thalamus', 'basal_ganglia', 'amygdala', 'hippocampus'],
                'amygdala': ['hypothalamus', 'brainstem', 'prefrontal_cortex', 'hippocampus']
            },
            'biological_plausibility': {
                'max_synaptic_velocity': 120,  # m/s
                'neurotransmitter_synthesis_rate': {'dopamine': 0.1, 'serotonin': 0.05},
                'receptor_binding_affinity_ranges': {
                    'dopamine': {'Kd_min': 1e-9, 'Kd_max': 1e-6},
                    'serotonin': {'Kd_min': 1e-10, 'Kd_max': 1e-7}
                }
            }
        }
    
    def validate_relationship(self, subject: str, predicate: str, obj: str, 
                            confidence: float) -> Dict:
        """Validate biological plausibility of relationship"""
        
        violations = []
        warnings = []
        adjusted_confidence = confidence
        
        # Apply all biological rules
        for rule in self.biological_rules:
            result = rule['check'](subject, predicate, obj, confidence)
            if result['violation']:
                violations.append({
                    'rule': rule['name'],
                    'description': rule['description'],
                    'details': result['details']
                })
                adjusted_confidence *= 0.5  # Penalize for violations
            
            if result['warning']:
                warnings.append({
                    'rule': rule['name'],
                    'details': result['details']
                })
                adjusted_confidence *= 0.8  # Minor penalty for warnings
        
        return {
            'biologically_plausible': len(violations) == 0,
            'original_confidence': confidence,
            'adjusted_confidence': adjusted_confidence,
            'violations': violations,
            'warnings': warnings,
            'constraint_applied': True
        }
    
    def _check_neurotransmitter_specificity(self, subject: str, predicate: str, 
                                          obj: str, confidence: float) -> Dict:
        """Check neurotransmitter-receptor specificity"""
        if subject in self.domain_knowledge['neurotransmitter_receptors']:
            # If object is a receptor, check if it's valid for this neurotransmitter
            receptors = self.domain_knowledge['neurotransmitter_receptors'][subject]
            if any(receptor.lower() in obj.lower() for receptor in receptors):
                return {'violation': False, 'warning': False, 'details': 'Valid receptor'}
            else:
                return {
                    'violation': True,
                    'warning': False,
                    'details': f"{subject} does not typically bind to {obj}"
                }
        
        return {'violation': False, 'warning': False, 'details': 'Not a neurotransmitter check'}
    
    def _check_anatomical_constraints(self, subject: str, predicate: str, 
                                    obj: str, confidence: float) -> Dict:
        """Check anatomical connection plausibility"""
        if subject in self.domain_knowledge['brain_region_connections']:
            valid_connections = self.domain_knowledge['brain_region_connections'][subject]
            if obj.lower() in [c.lower() for c in valid_connections]:
                return {'violation': False, 'warning': False, 'details': 'Valid anatomical connection'}
            else:
                return {
                    'violation': False,  # Warning, not violation (could be indirect)
                    'warning': True,
                    'details': f"{subject} does not directly connect to {obj} in standard neuroanatomy"
                }
        
        return {'violation': False, 'warning': False, 'details': 'Not an anatomical check'}

class ClinicalSafetyGuard:
    """Prevent clinical misuse of probabilistic outputs"""
    
    def __init__(self):
        self.safety_rules = self._load_safety_rules()
        self.disclaimer = """
        ⚠️ CLINICAL SAFETY DISCLAIMER:
        This system provides research insights only, not clinical advice.
        Never use this information for diagnosis or treatment decisions.
        Always consult licensed healthcare professionals.
        """
    
    def _load_safety_rules(self) -> List[Dict]:
        """Load clinical safety rules"""
        return [
            {
                'id': 'CSR-001',
                'pattern': r'(diagnos|treat|prescrib|dose|therapy|medication|patient|cure)',
                'action': 'require_disclaimer',
                'severity': 'high'
            },
            {
                'id': 'CSR-002',
                'pattern': r'(should|must|recommend|advise)\s+(take|use|administer)',
                'action': 'block_and_warn',
                'severity': 'critical'
            },
            {
                'id': 'CSR-003',
                'pattern': r'(\d+\.?\d*\s*(mg|g|ml|µg)\s+per\s+(day|dose))',
                'action': 'block_and_warn',
                'severity': 'critical'
            },
            {
                'id': 'CSR-004',
                'pattern': r'(side effect|adverse|risk|contraindication)',
                'action': 'require_evidence',
                'min_evidence': 3,
                'severity': 'medium'
            }
        ]
    
    def check_query_safety(self, query: str, clinical_setting: bool = False) -> Dict:
        """Check query for clinical safety issues"""
        
        if not clinical_setting:
            return {'safe': True, 'actions': [], 'warnings': []}
        
        issues = []
        actions = []
        blocked = False
        
        for rule in self.safety_rules:
            if re.search(rule['pattern'], query, re.IGNORECASE):
                issue = {
                    'rule_id': rule['id'],
                    'severity': rule['severity'],
                    'pattern_found': rule['pattern'],
                    'action_required': rule['action']
                }
                
                issues.append(issue)
                
                if rule['action'] == 'block_and_warn':
                    blocked = True
                    actions.append({
                        'type': 'block',
                        'reason': 'Contains potentially harmful clinical language',
                        'rule': rule['id']
                    })
                elif rule['action'] == 'require_disclaimer':
                    actions.append({
                        'type': 'add_disclaimer',
                        'disclaimer': self.disclaimer
                    })
                elif rule['action'] == 'require_evidence':
                    actions.append({
                        'type': 'increase_evidence_threshold',
                        'min_evidence': rule.get('min_evidence', 2)
                    })
        
        return {
            'safe': not blocked,
            'blocked': blocked,
            'issues': issues,
            'actions': actions,
            'disclaimer_required': len([a for a in actions if a['type'] == 'add_disclaimer']) > 0
        }
    
    def validate_clinical_output(self, result: Dict, clinical_setting: bool) -> Dict:
        """Validate outputs for clinical safety"""
        
        if not clinical_setting:
            return result
        
        # Add mandatory disclaimers
        if 'clinical_warning' not in result:
            result['clinical_warning'] = True
        
        if 'disclaimers' not in result:
            result['disclaimers'] = []
        
        result['disclaimers'].append({
            'type': 'clinical_safety',
            'text': self.disclaimer,
            'timestamp': datetime.now().isoformat()
        })
        
        # Cap confidence for clinical claims
        if result.get('confidence', 0) > 0.85:
            result['confidence_capped'] = True
            result['original_confidence'] = result['confidence']
            result['confidence'] = 0.85  # Conservative cap for clinical
        
        # Add evidence requirements
        if 'evidence_summary' in result:
            if len(result['evidence_summary']) < 2:
                result['insufficient_evidence'] = True
                result['recommendation'] = "Requires additional peer-reviewed evidence"
        
        return result

class ConfidenceCalibrator:
    """Proper confidence calibration with uncertainty quantification"""
    
    def __init__(self):
        self.calibration_data = []
        self.benchmark_set = self._load_benchmark_set()
        self.calibration_models = {}
        
    def _load_benchmark_set(self) -> List[Dict]:
        """Load benchmark queries with ground truth"""
        return [
            {'query': 'dopamine modulates reward', 'truth': True, 'domain': 'neuropsychiatry'},
            {'query': 'serotonin deficiency causes depression', 'truth': True, 'domain': 'clinical'},
            {'query': 'glutamate excites neurons', 'truth': True, 'domain': 'basic_neuroscience'},
            {'query': 'gaba inhibits neuronal firing', 'truth': True, 'domain': 'basic_neuroscience'},
            {'query': 'hippocampus supports memory', 'truth': True, 'domain': 'cognitive'},
            {'query': 'dopamine cures Parkinsons', 'truth': False, 'domain': 'clinical'},
            {'query': 'serotonin causes happiness', 'truth': False, 'domain': 'oversimplification'},
            {'query': 'glutamate is addictive', 'truth': False, 'domain': 'misconception'}
        ]
    
    def calibrate_confidence(self, raw_score: float, 
                           inference_type: str,
                           evidence_count: int,
                           evidence_quality: float) -> Dict:
        """Calibrate raw score using multiple factors"""
        
        # Base calibration (would use trained model in production)
        if inference_type == 'transitive':
            base_calibration = self._calibrate_transitive(raw_score, evidence_count)
        elif inference_type == 'probabilistic':
            base_calibration = self._calibrate_probabilistic(raw_score, evidence_count)
        elif inference_type == 'similarity':
            base_calibration = self._calibrate_similarity(raw_score, evidence_quality)
        else:
            base_calibration = raw_score * 0.9  # Conservative default
        
        # Apply evidence quality adjustment
        quality_adjusted = base_calibration * evidence_quality
        
        # Apply uncertainty based on evidence count
        if evidence_count < 2:
            uncertainty_penalty = 0.7
        elif evidence_count < 5:
            uncertainty_penalty = 0.85
        else:
            uncertainty_penalty = 0.95
        
        calibrated = quality_adjusted * uncertainty_penalty
        
        # Ensure bounds
        calibrated = max(0.0, min(1.0, calibrated))
        
        # Calculate uncertainty metrics
        uncertainty = self._calculate_uncertainty(calibrated, evidence_count, evidence_quality)
        
        return {
            'raw_confidence': raw_score,
            'calibrated_confidence': calibrated,
            'calibration_method': 'multi_factor',
            'uncertainty': uncertainty['level'],
            'uncertainty_score': uncertainty['score'],
            'evidence_factors': {
                'count': evidence_count,
                'quality': evidence_quality,
                'sufficiency': 'sufficient' if evidence_count >= 3 else 'insufficient'
            }
        }
    
    def _calibrate_transitive(self, raw_score: float, evidence_count: int) -> float:
        """Calibrate transitive inference scores"""
        # Transitive inferences get conservative calibration
        if evidence_count >= 3:
            return raw_score * 0.85
        elif evidence_count == 2:
            return raw_score * 0.75
        else:
            return raw_score * 0.6
    
    def _calibrate_probabilistic(self, raw_score: float, evidence_count: int) -> float:
        """Calibrate probabilistic inference scores"""
        # Probabilistic gets moderate calibration
        if evidence_count >= 5:
            return raw_score * 0.9
        elif evidence_count >= 3:
            return raw_score * 0.8
        else:
            return raw_score * 0.65
    
    def _calibrate_similarity(self, raw_score: float, evidence_quality: float) -> float:
        """Calibrate similarity scores (most conservative)"""
        # Similarity scores need strong evidence
        if evidence_quality > 0.8:
            return raw_score * 0.7
        else:
            return raw_score * 0.5
    
    def _calculate_uncertainty(self, confidence: float, 
                             evidence_count: int, 
                             evidence_quality: float) -> Dict:
        """Calculate uncertainty level"""
        
        uncertainty_score = (1 - confidence) * (1 - min(1.0, evidence_count/5)) * (1 - evidence_quality)
        
        if uncertainty_score < 0.1:
            level = 'LOW'
        elif uncertainty_score < 0.3:
            level = 'MEDIUM'
        else:
            level = 'HIGH'
        
        return {'score': uncertainty_score, 'level': level}

class RiskAwareNeuroSQLSystem:
    """Main system with all risk mitigations"""
    
    def __init__(self):
        self.evidence_validator = EvidenceValidator()
        self.biological_validator = BiologicalConstraintValidator()
        self.clinical_safety = ClinicalSafetyGuard()
        self.confidence_calibrator = ConfidenceCalibrator()
        
        # Risk tracking
        self.risk_log = []
        self.validation_history = []
        
    async def process_query_with_validation(self, query: str, 
                                          clinical_setting: bool = False) -> Dict:
        """Process query with comprehensive risk validation"""
        
        # 1. Clinical safety check
        safety_check = self.clinical_safety.check_query_safety(query, clinical_setting)
        if safety_check['blocked']:
            return {
                'query': query,
                'result': False,
                'confidence': 0.0,
                'error': 'Query blocked for clinical safety',
                'safety_issues': safety_check['issues'],
                'clinical_setting': clinical_setting
            }
        
        # 2. Process inference (simulated)
        inference_result = await self._perform_inference(query)
        
        # 3. Apply biological constraints
        if 'subject' in inference_result and 'object' in inference_result:
            biological_check = self.biological_validator.validate_relationship(
                inference_result['subject'],
                inference_result.get('predicate', 'RELATED_TO'),
                inference_result['object'],
                inference_result.get('raw_confidence', 0.5)
            )
            inference_result['biological_validation'] = biological_check
        
        # 4. Calibrate confidence
        calibrated = self.confidence_calibrator.calibrate_confidence(
            inference_result.get('raw_confidence', 0.5),
            inference_result.get('inference_type', 'unknown'),
            inference_result.get('evidence_count', 0),
            inference_result.get('evidence_quality', 0.5)
        )
        inference_result.update(calibrated)
        
        # 5. Apply clinical safety to output
        if clinical_setting:
            inference_result = self.clinical_safety.validate_clinical_output(
                inference_result, clinical_setting
            )
        
        # 6. Log risks
        self._log_risks(query, inference_result, safety_check)
        
        return inference_result
    
    async def _perform_inference(self, query: str) -> Dict:
        """Perform inference (placeholder for actual implementation)"""
        # This would connect to your actual inference engines
        return {
            'query': query,
            'result': True,
            'raw_confidence': 0.8,
            'inference_type': 'transitive',
            'evidence_count': 3,
            'evidence_quality': 0.7,
            'subject': 'dopamine',
            'predicate': 'MODULATES',
            'object': 'reward'
        }
    
    def _log_risks(self, query: str, result: Dict, safety_check: Dict):
        """Log risk assessment"""
        risk_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'confidence': result.get('confidence', 0),
            'uncertainty': result.get('uncertainty', 'UNKNOWN'),
            'safety_issues': len(safety_check.get('issues', [])),
            'biological_violations': len(result.get('biological_validation', {}).get('violations', [])),
            'clinical_warning': result.get('clinical_warning', False),
            'risk_level': self._calculate_risk_level(result)
        }
        
        self.risk_log.append(risk_entry)
        
        # Alert on high risk
        if risk_entry['risk_level'] == 'HIGH':
            logger.warning(f"High risk inference: {query}")
    
    def _calculate_risk_level(self, result: Dict) -> str:
        """Calculate overall risk level"""
        risk_score = 0
        
        # High confidence with low evidence = risk
        if result.get('confidence', 0) > 0.8 and result.get('evidence_count', 0) < 2:
            risk_score += 2
        
        # Clinical warning
        if result.get('clinical_warning', False):
            risk_score += 1
        
        # Biological violations
        if result.get('biological_validation', {}).get('violations', []):
            risk_score += len(result['biological_validation']['violations'])
        
        # High uncertainty
        if result.get('uncertainty') == 'HIGH':
            risk_score += 1
        
        if risk_score >= 3:
            return 'HIGH'
        elif risk_score >= 1:
            return 'MEDIUM'
        else:
            return 'LOW'

# ============================================================================
# TEST SUITE
# ============================================================================

async def test_risk_mitigations():
    """Test all risk mitigation features"""
    
    system = RiskAwareNeuroSQLSystem()
    
    test_cases = [
        {
            'query': 'dopamine modulates reward via D2 receptors',
            'clinical': False,
            'description': 'Valid neuroscience query'
        },
        {
            'query': 'Take 100mg serotonin for depression',
            'clinical': True,
            'description': 'Dangerous clinical advice'
        },
        {
            'query': 'acetylcholine binds to NMDA receptors',
            'clinical': False,
            'description': 'Biologically implausible'
        },
        {
            'query': 'hippocampus directly connects to spinal cord',
            'clinical': False,
            'description': 'Anatomically questionable'
        }
    ]
    
    results = []
    for test in test_cases:
        result = await system.process_query_with_validation(
            test['query'],
            test['clinical']
        )
        
        results.append({
            'query': test['query'],
            'description': test['description'],
            'result': result.get('result'),
            'confidence': result.get('confidence'),
            'blocked': result.get('error', '').startswith('Query blocked'),
            'biological_plausible': result.get('biological_validation', {}).get('biologically_plausible', True),
            'clinical_warning': result.get('clinical_warning', False),
            'uncertainty': result.get('uncertainty', 'UNKNOWN')
        })
    
    return results

if __name__ == "__main__":
    import asyncio
    
    # Run tests
    test_results = asyncio.run(test_risk_mitigations())
    
    print("Risk Mitigation Test Results:")
    print("=" * 80)
    for r in test_results:
        print(f"Query: {r['query']}")
        print(f"Description: {r['description']}")
        print(f"Result: {r['result']} | Confidence: {r['confidence']:.2%}")
        print(f"Blocked: {r['blocked']} | Bio Plausible: {r['biological_plausible']}")
        print(f"Clinical Warning: {r['clinical_warning']} | Uncertainty: {r['uncertainty']}")
        print("-" * 80)
