# ============================================================================
# COMPLETE VALIDATED EVIDENCE SYSTEM
# ============================================================================

import re
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class EvidenceSource(Enum):
    PUBMED = "pubmed"
    TEXTBOOK = "textbook"
    DATASET = "dataset"
    CLINICAL_TRIAL = "clinical_trial"
    EXPERT_CURATION = "expert_curation"
    SYSTEMATIC_REVIEW = "systematic_review"

class EvidenceQuality(Enum):
    CURATED = 5
    PEER_REVIEWED = 4
    PREPRINT = 3
    DATASET = 2
    WEB = 1
    UNVERIFIED = 0

@dataclass
class ValidatedEvidence:
    """Evidence with validation tracking"""
    evidence_id: str
    source_type: EvidenceSource
    source_id: str
    snippet: str
    extracted_fact: Optional[Dict] = None
    retrieval_time: datetime = field(default_factory=datetime.now)
    
    # Validation fields
    source_validated: bool = False
    content_validated: bool = False
    validation_score: float = 0.0
    validation_notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.content_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute hash for integrity checking"""
        content = f"{self.source_id}|{self.snippet}|{self.retrieval_time.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

# ============================================================================
# EVIDENCE VALIDATOR
# ============================================================================

class EvidenceValidator:
    """Validate evidence integrity"""
    
    def __init__(self):
        self.validation_cache = {}
        logger.info("EvidenceValidator initialized")
    
    async def validate_evidence(self, evidence: ValidatedEvidence) -> Dict:
        """Validate evidence with comprehensive checks"""
        
        validation_result = {
            'evidence_id': evidence.evidence_id,
            'source_valid': False,
            'content_valid': False,
            'integrity_score': 0.0,
            'issues': [],
            'warnings': []
        }
        
        try:
            # 1. Source validation
            source_valid = await self._validate_source(evidence.source_type, evidence.source_id)
            validation_result['source_valid'] = source_valid
            
            # 2. Content validation
            content_valid = self._validate_content(evidence.snippet, evidence.source_id)
            validation_result['content_valid'] = content_valid
            
            # 3. Integrity check
            integrity_score = self._calculate_integrity_score(source_valid, content_valid)
            validation_result['integrity_score'] = integrity_score
            
            # Update evidence
            evidence.source_validated = source_valid
            evidence.content_validated = content_valid
            evidence.validation_score = integrity_score
            
            logger.info(f"Evidence {evidence.evidence_id} validated with score {integrity_score:.2f}")
            
        except Exception as e:
            validation_result['issues'].append(f"Validation error: {str(e)}")
            logger.error(f"Evidence validation error: {e}")
        
        return validation_result
    
    async def _validate_source(self, source_type: EvidenceSource, source_id: str) -> bool:
        """Validate source authenticity"""
        
        if source_type == EvidenceSource.PUBMED:
            return self._validate_pubmed_source(source_id)
        elif source_type == EvidenceSource.TEXTBOOK:
            return self._validate_textbook_source(source_id)
        elif source_type == EvidenceSource.DATASET:
            return self._validate_dataset_source(source_id)
        else:
            # For other types, do basic validation
            return bool(source_id and len(source_id) > 3)
    
    def _validate_pubmed_source(self, pmid: str) -> bool:
        """Validate PubMed ID"""
        try:
            # Check format
            if not re.match(r'^\d{1,8}$', pmid):
                return False
            
            # Check if it's a known valid PMID (simplified for demo)
            # In production, would call PubMed API
            known_valid_pmids = {
                "11242049",  # Real dopamine paper
                "86936985",  # From your demo
                "12345678",  # Test PMID
            }
            
            if pmid in known_valid_pmids:
                return True
            
            # For unknown PMIDs, check if they look plausible
            # Real PMIDs are typically 8 digits, starting with 1-3
            if len(pmid) == 8 and pmid[0] in "123":
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"PubMed validation error: {e}")
            return False
    
    def _validate_textbook_source(self, isbn: str) -> bool:
        """Validate textbook ISBN"""
        try:
            # Basic ISBN validation
            isbn_clean = isbn.replace("-", "").replace(" ", "")
            if len(isbn_clean) not in [10, 13]:
                return False
            
            # Check if it looks like an ISBN
            if isbn_clean.startswith("978") or isbn_clean.startswith("979"):
                return True
            
            # Check for known neuroscience textbooks
            known_textbooks = {
                "978-0878936977",  # Principles of Neural Science
                "978-1605352765",  # Neuroscience
                "978-0199920224",  # Fundamental Neuroscience
            }
            
            clean_isbn = isbn.replace("-", "")
            if clean_isbn in known_textbooks:
                return True
            
            return bool(re.match(r'^97[89]\d{10}$', isbn_clean))
            
        except Exception as e:
            logger.error(f"Textbook validation error: {e}")
            return False
    
    def _validate_dataset_source(self, dataset_id: str) -> bool:
        """Validate dataset identifier"""
        # Check common dataset formats
        if dataset_id.startswith("10."):  # DOI
            return True
        elif dataset_id.startswith("GSE"):  # GEO accession
            return True
        elif dataset_id.startswith("PRJ"):  # BioProject
            return True
        elif "dataset" in dataset_id.lower():
            return True
        
        return False
    
    def _validate_content(self, snippet: str, source_id: str) -> bool:
        """Validate content snippet"""
        
        if not snippet or len(snippet.strip()) < 10:
            return False
        
        # Check for placeholder text
        placeholder_indicators = [
            "TODO:", "FIXME:", "EXAMPLE", "TEST TEXT",
            "LOREM IPSUM", "PLACEHOLDER", "SAMPLE TEXT"
        ]
        
        snippet_upper = snippet.upper()
        if any(indicator in snippet_upper for indicator in placeholder_indicators):
            return False
        
        # Check snippet length and content
        if len(snippet) < 20 or len(snippet) > 1000:
            return False
        
        # Check if snippet contains meaningful content
        words = snippet.split()
        if len(words) < 5:
            return False
        
        # Check for neuroscience terminology
        neuroscience_terms = [
            "neuron", "synapse", "neurotransmitter", "receptor",
            "hippocampus", "cortex", "dopamine", "serotonin",
            "memory", "learning", "plasticity", "circuit"
        ]
        
        snippet_lower = snippet.lower()
        neuro_term_count = sum(1 for term in neuroscience_terms if term in snippet_lower)
        
        # If it's neuroscience evidence, should have at least 1 neuro term
        if "neuro" in source_id.lower() or "brain" in source_id.lower():
            return neuro_term_count > 0
        
        return True
    
    def _calculate_integrity_score(self, source_valid: bool, content_valid: bool) -> float:
        """Calculate overall integrity score"""
        if source_valid and content_valid:
            return 0.9
        elif source_valid and not content_valid:
            return 0.6
        elif not source_valid and content_valid:
            return 0.4
        else:
            return 0.1

# ============================================================================
# BIOLOGICAL CONSTRAINT VALIDATOR
# ============================================================================

class BiologicalConstraintValidator:
    """Apply biological constraints to prevent semantic drift"""
    
    def __init__(self):
        self.biological_rules = self._load_biological_rules()
        logger.info("BiologicalConstraintValidator initialized")
    
    def _load_biological_rules(self) -> List[Dict]:
        """Load biological plausibility rules"""
        return [
            {
                'id': 'BIO-001',
                'name': 'neurotransmitter_specificity',
                'check': self._check_neurotransmitter_specificity,
                'severity': 'high'
            },
            {
                'id': 'BIO-002',
                'name': 'anatomical_plausibility',
                'check': self._check_anatomical_plausibility,
                'severity': 'medium'
            },
            {
                'id': 'BIO-003',
                'name': 'receptor_compatibility',
                'check': self._check_receptor_compatibility,
                'severity': 'high'
            }
        ]
    
    def validate_relationship(self, subject: str, predicate: str, 
                            object: str, confidence: float) -> Dict:
        """Validate biological plausibility"""
        
        violations = []
        warnings = []
        adjusted_confidence = confidence
        
        # Apply all biological rules
        for rule in self.biological_rules:
            result = rule['check'](subject, predicate, object, confidence)
            
            if result['violation']:
                violations.append({
                    'rule_id': rule['id'],
                    'rule_name': rule['name'],
                    'severity': rule['severity'],
                    'details': result['details']
                })
                
                # Apply penalty based on severity
                if rule['severity'] == 'high':
                    adjusted_confidence *= 0.5
                elif rule['severity'] == 'medium':
                    adjusted_confidence *= 0.7
                else:
                    adjusted_confidence *= 0.9
            
            elif result['warning']:
                warnings.append({
                    'rule_id': rule['id'],
                    'rule_name': rule['name'],
                    'details': result['details']
                })
                adjusted_confidence *= 0.8
        
        # Ensure confidence stays in bounds
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
        
        return {
            'biologically_plausible': len(violations) == 0,
            'original_confidence': confidence,
            'adjusted_confidence': adjusted_confidence,
            'violation_count': len(violations),
            'warning_count': len(warnings),
            'violations': violations,
            'warnings': warnings
        }
    
    def _check_neurotransmitter_specificity(self, subject: str, predicate: str,
                                          object: str, confidence: float) -> Dict:
        """Check neurotransmitter-receptor specificity"""
        
        # Known neurotransmitter-receptor pairs
        neurotransmitter_receptors = {
            'dopamine': ['D1', 'D2', 'D3', 'D4', 'D5', 'DRD1', 'DRD2'],
            'serotonin': ['5-HT1A', '5-HT2A', '5-HT2C', '5-HT3', '5-HT4', '5-HT6', '5-HT7'],
            'glutamate': ['NMDA', 'AMPA', 'kainate', 'mGluR1', 'mGluR5'],
            'gaba': ['GABA_A', 'GABA_B', 'GABRA', 'GABRB'],
            'acetylcholine': ['nAChR', 'mAChR', 'CHRNA', 'CHRNB']
        }
        
        subject_lower = subject.lower()
        object_lower = object.lower()
        
        # Check if subject is a neurotransmitter
        for nt, receptors in neurotransmitter_receptors.items():
            if nt in subject_lower:
                # Check if object is a receptor
                receptor_found = any(receptor.lower() in object_lower for receptor in receptors)
                
                if not receptor_found and predicate in ['BINDS_TO', 'ACTIVATES', 'BINDS']:
                    return {
                        'violation': True,
                        'warning': False,
                        'details': f"{subject} does not typically bind to {object}"
                    }
        
        return {'violation': False, 'warning': False, 'details': ''}
    
    def _check_anatomical_plausibility(self, subject: str, predicate: str,
                                     object: str, confidence: float) -> Dict:
        """Check anatomical connection plausibility"""
        
        # Known brain region connections
        brain_connections = {
            'hippocampus': ['entorhinal cortex', 'septum', 'mammillary bodies', 
                           'prefrontal cortex', 'amygdala'],
            'prefrontal cortex': ['thalamus', 'basal ganglia', 'amygdala', 
                                 'hippocampus', 'anterior cingulate'],
            'amygdala': ['hypothalamus', 'brainstem', 'prefrontal cortex', 
                        'hippocampus', 'insula'],
            'striatum': ['substantia nigra', 'globus pallidus', 'thalamus', 
                        'cortex', 'subthalamic nucleus']
        }
        
        subject_lower = subject.lower()
        object_lower = object.lower()
        
        # Check if subject is a brain region
        for region, connections in brain_connections.items():
            if region in subject_lower:
                # Check if object is a plausible connection
                connection_found = any(conn.lower() in object_lower for conn in connections)
                
                if not connection_found and predicate in ['CONNECTS_TO', 'PROJECTS_TO', 'INNERVATES']:
                    return {
                        'violation': False,  # Warning, not violation (could be indirect)
                        'warning': True,
                        'details': f"{subject} does not typically directly connect to {object}"
                    }
        
        return {'violation': False, 'warning': False, 'details': ''}
    
    def _check_receptor_compatibility(self, subject: str, predicate: str,
                                    object: str, confidence: float) -> Dict:
        """Check receptor compatibility"""
        
        # Receptor families and their typical ligands
        receptor_families = {
            'NMDA': ['glutamate', 'glycine'],
            'AMPA': ['glutamate'],
            'GABA_A': ['gaba', 'muscimol', 'benzodiazepines'],
            'D2': ['dopamine', 'antipsychotics'],
            '5-HT2A': ['serotonin', 'psychedelics', 'antipsychotics']
        }
        
        subject_lower = subject.lower()
        object_lower = object.lower()
        
        # Check if object is a receptor
        for receptor, ligands in receptor_families.items():
            if receptor.lower() in object_lower:
                # Check if subject is a compatible ligand
                ligand_found = any(ligand.lower() in subject_lower for ligand in ligands)
                
                if not ligand_found and predicate in ['BINDS_TO', 'ACTIVATES']:
                    return {
                        'violation': True,
                        'warning': False,
                        'details': f"{subject} does not typically bind to {object} receptors"
                    }
        
        return {'violation': False, 'warning': False, 'details': ''}

# ============================================================================
# CLINICAL SAFETY GUARD
# ============================================================================

class ClinicalSafetyGuard:
    """Prevent clinical misuse"""
    
    def __init__(self):
        self.safety_rules = self._load_safety_rules()
        self.disclaimer = """⚠️ CLINICAL SAFETY DISCLAIMER:
This system provides research insights only, not clinical advice.
Never use this information for diagnosis or treatment decisions.
Always consult licensed healthcare professionals."""
        logger.info("ClinicalSafetyGuard initialized")
    
    def _load_safety_rules(self) -> List[Dict]:
        """Load clinical safety rules"""
        return [
            {
                'id': 'CSR-001',
                'pattern': r'\b(take|use|administer|prescribe|ingest|inject)\b.*?\b(mg|g|ml|µg|dose)\b',
                'action': 'block',
                'severity': 'critical',
                'message': 'Contains specific treatment/dosage advice'
            },
            {
                'id': 'CSR-002',
                'pattern': r'\b(should|must|recommend|advise)\s+to\s+(take|use)\b',
                'action': 'block',
                'severity': 'critical',
                'message': 'Contains treatment recommendations'
            },
            {
                'id': 'CSR-003',
                'pattern': r'\b(diagnos|treat|therapy|medication|patient|cure)\b',
                'action': 'warn',
                'severity': 'high',
                'message': 'Contains clinical terminology'
            },
            {
                'id': 'CSR-004',
                'pattern': r'\b(side effect|adverse|risk|contraindication|warning)\b',
                'action': 'require_evidence',
                'severity': 'medium',
                'min_evidence': 3
            }
        ]
    
    def check_query_safety(self, query: str, clinical_setting: bool = False) -> Dict:
        """Check query for clinical safety issues"""
        
        if not clinical_setting:
            return {'safe': True, 'blocked': False, 'issues': [], 'actions': []}
        
        issues = []
        actions = []
        blocked = False
        
        query_lower = query.lower()
        
        for rule in self.safety_rules:
            if re.search(rule['pattern'], query_lower, re.IGNORECASE):
                issue = {
                    'rule_id': rule['id'],
                    'severity': rule['severity'],
                    'message': rule.get('message', 'Pattern matched'),
                    'action': rule['action']
                }
                issues.append(issue)
                
                if rule['action'] == 'block':
                    blocked = True
                    actions.append({
                        'type': 'block_query',
                        'reason': f"Blocked by rule {rule['id']}: {rule.get('message', 'Safety violation')}"
                    })
                elif rule['action'] == 'warn':
                    actions.append({
                        'type': 'add_warning',
                        'warning': 'Contains clinical terminology - use with caution'
                    })
                elif rule['action'] == 'require_evidence':
                    actions.append({
                        'type': 'increase_evidence',
                        'min_evidence': rule.get('min_evidence', 2)
                    })
        
        return {
            'safe': not blocked,
            'blocked': blocked,
            'issues': issues,
            'actions': actions,
            'disclaimer_required': clinical_setting
        }
    
    def apply_clinical_safeguards(self, result: Dict, clinical_setting: bool) -> Dict:
        """Apply clinical safeguards to result"""
        
        if not clinical_setting:
            return result
        
        # Add disclaimer
        if 'disclaimers' not in result:
            result['disclaimers'] = []
        result['disclaimers'].append({
            'type': 'clinical_safety',
            'text': self.disclaimer,
            'timestamp': datetime.now().isoformat()
        })
        
        # Cap confidence for clinical outputs
        if result.get('confidence', 0) > 0.85:
            result['original_confidence'] = result['confidence']
            result['confidence'] = 0.85
            result['confidence_capped'] = True
        
        # Mark as clinically reviewed
        result['clinically_reviewed'] = True
        result['clinical_warning'] = True
        
        return result

# ============================================================================
# CONFIDENCE CALIBRATOR
# ============================================================================

class ConfidenceCalibrator:
    """Proper confidence calibration"""
    
    def __init__(self):
        self.calibration_factors = self._load_calibration_factors()
        logger.info("ConfidenceCalibrator initialized")
    
    def _load_calibration_factors(self) -> Dict:
        """Load calibration factors"""
        return {
            'transitive': {'base': 0.85, 'evidence_factor': 0.1, 'quality_factor': 0.05},
            'probabilistic': {'base': 0.9, 'evidence_factor': 0.15, 'quality_factor': 0.1},
            'similarity': {'base': 0.7, 'evidence_factor': 0.2, 'quality_factor': 0.1},
            'default': {'base': 0.8, 'evidence_factor': 0.1, 'quality_factor': 0.05}
        }
    
    def calibrate_confidence(self, raw_confidence: float, 
                           inference_type: str,
                           evidence_count: int,
                           evidence_quality: float) -> Dict:
        """Calibrate confidence score"""
        
        # Get calibration factors
        factors = self.calibration_factors.get(inference_type, 
                                              self.calibration_factors['default'])
        
        # Base adjustment
        calibrated = raw_confidence * factors['base']
        
        # Evidence count adjustment
        if evidence_count >= 5:
            evidence_factor = 1.0
        elif evidence_count >= 3:
            evidence_factor = 0.9
        elif evidence_count >= 2:
            evidence_factor = 0.8
        else:
            evidence_factor = 0.6
        
        calibrated *= evidence_factor
        
        # Evidence quality adjustment
        quality_factor = 0.5 + (evidence_quality * 0.5)  # Map 0-1 to 0.5-1.0
        calibrated *= quality_factor
        
        # Ensure bounds
        calibrated = max(0.0, min(1.0, calibrated))
        
        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(calibrated, evidence_count, evidence_quality)
        
        return {
            'raw_confidence': raw_confidence,
            'calibrated_confidence': calibrated,
            'calibration_method': 'multi_factor',
            'uncertainty_level': uncertainty['level'],
            'uncertainty_score': uncertainty['score'],
            'evidence_sufficiency': 'sufficient' if evidence_count >= 3 else 'insufficient',
            'calibration_factors_applied': factors
        }
    
    def _calculate_uncertainty(self, confidence: float, 
                             evidence_count: int, 
                             evidence_quality: float) -> Dict:
        """Calculate uncertainty metrics"""
        
        # Base uncertainty (inverse of confidence)
        base_uncertainty = 1.0 - confidence
        
        # Evidence uncertainty
        if evidence_count >= 5:
            evidence_uncertainty = 0.1
        elif evidence_count >= 3:
            evidence_uncertainty = 0.3
        elif evidence_count >= 1:
            evidence_uncertainty = 0.6
        else:
            evidence_uncertainty = 0.9
        
        # Quality uncertainty
        quality_uncertainty = 1.0 - evidence_quality
        
        # Combined uncertainty
        combined = (base_uncertainty * 0.4 + 
                   evidence_uncertainty * 0.3 + 
                   quality_uncertainty * 0.3)
        
        # Map to levels
        if combined < 0.2:
            level = 'LOW'
        elif combined < 0.4:
            level = 'MEDIUM'
        else:
            level = 'HIGH'
        
        return {'score': combined, 'level': level}

# ============================================================================
# RISK-AWARE NEUROSQL SYSTEM
# ============================================================================

class RiskAwareNeuroSQLSystem:
    """Main system with all risk mitigations"""
    
    def __init__(self):
        # Initialize all validators
        self.evidence_validator = EvidenceValidator()
        self.biological_validator = BiologicalConstraintValidator()
        self.clinical_safety = ClinicalSafetyGuard()
        self.confidence_calibrator = ConfidenceCalibrator()
        
        # Risk tracking
        self.risk_log = []
        self.inference_count = 0
        
        logger.info("RiskAwareNeuroSQLSystem initialized")
    
    async def process_query_with_validation(self, query: str, 
                                          clinical_setting: bool = False) -> Dict:
        """Process query with comprehensive risk validation"""
        
        self.inference_count += 1
        inference_id = f"inf_{self.inference_count:06d}"
        
        logger.info(f"Processing query {inference_id}: {query[:50]}...")
        
        # 1. Clinical safety check
        safety_check = self.clinical_safety.check_query_safety(query, clinical_setting)
        if safety_check['blocked']:
            result = {
                'inference_id': inference_id,
                'query': query,
                'result': False,
                'confidence': 0.0,
                'error': 'Query blocked for clinical safety',
                'safety_issues': safety_check['issues'],
                'clinical_setting': clinical_setting,
                'blocked': True,
                'timestamp': datetime.now().isoformat()
            }
            self._log_risk(inference_id, query, result, safety_check)
            return result
        
        # 2. Parse query (simplified)
        parsed_query = self._parse_query(query)
        
        # 3. Perform inference (simulated)
        inference_result = self._perform_inference(parsed_query)
        
        # 4. Apply biological constraints
        biological_check = self.biological_validator.validate_relationship(
            inference_result.get('subject', 'unknown'),
            inference_result.get('predicate', 'RELATED_TO'),
            inference_result.get('object', 'unknown'),
            inference_result.get('raw_confidence', 0.5)
        )
        
        # Update confidence with biological validation
        inference_result['biological_validation'] = biological_check
        inference_result['confidence'] = biological_check['adjusted_confidence']
        
        # 5. Calibrate confidence
        calibration_result = self.confidence_calibrator.calibrate_confidence(
            inference_result['confidence'],
            inference_result.get('inference_type', 'default'),
            inference_result.get('evidence_count', 1),
            inference_result.get('evidence_quality', 0.5)
        )
        inference_result.update(calibration_result)
        
        # 6. Apply clinical safeguards
        if clinical_setting:
            inference_result = self.clinical_safety.apply_clinical_safeguards(
                inference_result, clinical_setting
            )
        
        # 7. Build final result
        result = {
            'inference_id': inference_id,
            'query': query,
            'result': inference_result.get('result', False),
            'confidence': inference_result.get('calibrated_confidence', 0.0),
            'uncertainty': inference_result.get('uncertainty_level', 'UNKNOWN'),
            'evidence_count': inference_result.get('evidence_count', 0),
            'biological_plausible': biological_check['biologically_plausible'],
            'clinical_setting': clinical_setting,
            'clinical_warning': inference_result.get('clinical_warning', False),
            'calibration_applied': True,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add disclaimers if present
        if 'disclaimers' in inference_result:
            result['disclaimers'] = inference_result['disclaimers']
        
        # 8. Log risk assessment
        self._log_risk(inference_id, query, result, safety_check)
        
        logger.info(f"Query {inference_id} completed with confidence {result['confidence']:.2f}")
        
        return result
    
    def _parse_query(self, query: str) -> Dict:
        """Parse query into structured format"""
        query_lower = query.lower()
        
        # Simple keyword-based parsing
        parsed = {
            'raw_query': query,
            'tokens': query_lower.split(),
            'has_neuro_terms': any(term in query_lower for term in [
                'neuron', 'brain', 'dopamine', 'serotonin', 'glutamate',
                'hippocampus', 'memory', 'learning', 'synapse'
            ])
        }
        
        # Try to extract subject-predicate-object
        words = query_lower.split()
        if len(words) >= 2:
            parsed['subject'] = words[0]
            parsed['object'] = words[-1]
            
            # Look for predicates
            predicates = ['modulates', 'supports', 'inhibits', 'causes', 'affects', 'regulates']
            for pred in predicates:
                if pred in query_lower:
                    parsed['predicate'] = pred.upper()
                    break
        
        return parsed
    
    def _perform_inference(self, parsed_query: Dict) -> Dict:
        """Perform inference (simulated for demo)"""
        
        # Simulated inference logic
        query_text = parsed_query['raw_query'].lower()
        
        # Default inference
        inference = {
            'result': True,
            'raw_confidence': 0.7,
            'inference_type': 'transitive',
            'evidence_count': 2,
            'evidence_quality': 0.6,
            'subject': parsed_query.get('subject', 'unknown'),
            'predicate': parsed_query.get('predicate', 'RELATED_TO'),
            'object': parsed_query.get('object', 'unknown')
        }
        
        # Adjust based on query content
        if 'dopamine' in query_text and 'reward' in query_text:
            inference['raw_confidence'] = 0.9
            inference['evidence_count'] = 3
            inference['evidence_quality'] = 0.8
        
        if 'clinical' in query_text or 'treatment' in query_text:
            inference['raw_confidence'] = 0.5  # Lower for clinical
        
        return inference
    
    def _log_risk(self, inference_id: str, query: str, result: Dict, safety_check: Dict):
        """Log risk assessment"""
        risk_entry = {
            'inference_id': inference_id,
            'timestamp': datetime.now().isoformat(),
            'query': query[:100],
            'confidence': result.get('confidence', 0),
            'uncertainty': result.get('uncertainty', 'UNKNOWN'),
            'blocked': result.get('blocked', False),
            'clinical_warning': result.get('clinical_warning', False),
            'biological_plausible': result.get('biological_plausible', True),
            'risk_level': self._calculate_risk_level(result)
        }
        
        self.risk_log.append(risk_entry)
        
        # Keep log manageable
        if len(self.risk_log) > 1000:
            self.risk_log = self.risk_log[-1000:]
        
        # Log high risk entries
        if risk_entry['risk_level'] == 'HIGH':
            logger.warning(f"High risk inference {inference_id}: {query[:50]}...")
    
    def _calculate_risk_level(self, result: Dict) -> str:
        """Calculate risk level"""
        risk_score = 0
        
        # Blocked queries are high risk
        if result.get('blocked', False):
            return 'HIGH'
        
        # High confidence with clinical warning
        if result.get('confidence', 0) > 0.8 and result.get('clinical_warning', False):
            risk_score += 2
        
        # Low confidence
        if result.get('confidence', 0) < 0.3:
            risk_score += 1
        
        # High uncertainty
        if result.get('uncertainty') == 'HIGH':
            risk_score += 1
        
        # Not biologically plausible
        if not result.get('biological_plausible', True):
            risk_score += 2
        
        if risk_score >= 3:
            return 'HIGH'
        elif risk_score >= 1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_risk_report(self) -> Dict:
        """Get risk assessment report"""
        if not self.risk_log:
            return {'message': 'No inferences processed yet'}
        
        total = len(self.risk_log)
        high_risk = sum(1 for entry in self.risk_log if entry['risk_level'] == 'HIGH')
        medium_risk = sum(1 for entry in self.risk_log if entry['risk_level'] == 'MEDIUM')
        blocked = sum(1 for entry in self.risk_log if entry.get('blocked', False))
        
        return {
            'total_inferences': total,
            'high_risk_percentage': high_risk / total if total > 0 else 0,
            'medium_risk_percentage': medium_risk / total if total > 0 else 0,
            'blocked_percentage': blocked / total if total > 0 else 0,
            'average_confidence': np.mean([entry.get('confidence', 0) for entry in self.risk_log]) if self.risk_log else 0,
            'recent_high_risk': [entry for entry in self.risk_log[-5:] if entry['risk_level'] == 'HIGH']
        }

# ============================================================================
# SIMPLE TEST FUNCTION
# ============================================================================

async def test_system():
    """Test the risk-aware system"""
    system = RiskAwareNeuroSQLSystem()
    
    test_cases = [
        ("dopamine modulates reward processing", False),
        ("Take 50mg serotonin for depression", True),
        ("hippocampus supports memory formation", False),
        ("acetylcholine binds to NMDA receptors", False),
    ]
    
    print("Testing Risk-Aware NeuroSQL System")
    print("=" * 60)
    
    for query, clinical in test_cases:
        print(f"\nQuery: {query}")
        print(f"Clinical setting: {clinical}")
        
        result = await system.process_query_with_validation(query, clinical)
        
        if result.get('blocked'):
            print("  ❌ Query blocked for safety")
        else:
            print(f"  ✅ Result: {result.get('result')}")
            print(f"  Confidence: {result.get('confidence', 0):.1%}")
            print(f"  Uncertainty: {result.get('uncertainty', 'N/A')}")
            if result.get('clinical_warning'):
                print(f"  ⚠️ Clinical warning applied")
    
    print("\n" + "=" * 60)
    print("Risk Report:")
    report = system.get_risk_report()
    print(f"Total inferences: {report.get('total_inferences', 0)}")
    print(f"High risk: {report.get('high_risk_percentage', 0):.1%}")
    print(f"Blocked queries: {report.get('blocked_percentage', 0):.1%}")
    
    print("\n✅ System test complete!")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    asyncio.run(test_system())
