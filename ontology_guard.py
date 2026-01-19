# ontology_guard.py
"""
Ontology Guard - Prevents cross-domain semantic contamination
"""

class OntologyDomain:
    """Define ontology domains and their relationships"""
    
    BIOLOGY = {
        'name': 'biology',
        'keywords': {'animal', 'mammal', 'dog', 'cat', 'pet', 'biology', 'organism', 'living', 'creature'},
        'types': {'class', 'instance'},
        'compatible_with': ['abstract'],
        'abstract_parent': 'LivingBeing'
    }
    
    MACHINE_LEARNING = {
        'name': 'machine_learning',
        'keywords': {'neural', 'deep', 'learning', 'cnn', 'rnn', 'machine', 'ai', 'network', 'model'},
        'types': {'class'},
        'compatible_with': ['computer_science', 'abstract'],
        'abstract_parent': 'Algorithm'
    }
    
    COMPUTER_SCIENCE = {
        'name': 'computer_science',
        'keywords': {'software', 'programming', 'algorithm', 'code', 'computer', 'system', 'data'},
        'types': {'class'},
        'compatible_with': ['abstract', 'machine_learning'],
        'abstract_parent': 'Concept'
    }
    
    INSTANCE = {
        'name': 'instance',
        'types': {'instance'},
        'detection': 'proper_noun',  # Names like Fluffy, Fido
        'compatible_with': []  # Only compatible with immediate class
    }
    
    ABSTRACT = {
        'name': 'abstract',
        'keywords': {'concept', 'entity', 'object', 'thing', 'being', 'idea'},
        'types': {'class'},
        'compatible_with': ['all'],  # Abstracts connect to everything
        'abstract_parent': None
    }
    
    ALL_DOMAINS = [BIOLOGY, MACHINE_LEARNING, COMPUTER_SCIENCE, ABSTRACT, INSTANCE]

class OntologyGuard:
    """Main guard class to prevent invalid inferences"""
    
    def __init__(self):
        self.domains = OntologyDomain.ALL_DOMAINS
        self.violations = []
        self.warnings = []
    
    def get_domain(self, concept_name):
        """Determine which domain a concept belongs to"""
        
        # Check if it's an instance (proper noun)
        if self._is_proper_noun(concept_name):
            return OntologyDomain.INSTANCE
        
        concept_lower = concept_name.lower()
        
        # Check each domain
        for domain in self.domains:
            if domain['name'] == 'instance':
                continue
                
            if 'keywords' in domain:
                for keyword in domain['keywords']:
                    if keyword in concept_lower:
                        return domain
        
        # Default to abstract
        return OntologyDomain.ABSTRACT
    
    def _is_proper_noun(self, name):
        """Check if a name is a proper noun (instance)"""
        # Rule: Starts with capital, rest lowercase, not a common word
        if (len(name) > 1 and 
            name[0].isupper() and 
            name[1:].islower() and
            name.lower() not in {'python', 'software', 'deeplearning'}):
            return True
        return False
    
    def validate_relationship(self, subject, relation, object, confidence=1.0):
        """Validate if a relationship is semantically valid"""
        
        subj_domain = self.get_domain(subject)
        obj_domain = self.get_domain(object)
        
        # Get domain names
        subj_name = subj_domain['name']
        obj_name = obj_domain['name']
        
        # CRITICAL RULE 1: No cross-domain is_a (except through abstract)
        if relation == 'is_a':
            if subj_name != obj_name:
                # Check if they're connected through abstract
                if ('abstract' not in [subj_name, obj_name] and
                    'all' not in subj_domain.get('compatible_with', []) and
                    'all' not in obj_domain.get('compatible_with', [])):
                    
                    reason = f"Cross-domain is_a: {subj_name} → {obj_name}"
                    self.violations.append({
                        'type': 'cross_domain_is_a',
                        'subject': subject,
                        'object': object,
                        'relation': relation,
                        'reason': reason,
                        'confidence': confidence
                    })
                    return False, reason
            
            # CRITICAL RULE 2: Instances can only instance_of, not is_a
            if subj_name == 'instance' and relation == 'is_a':
                reason = f"Instance {subject} cannot use is_a (use instance_of)"
                self.violations.append({
                    'type': 'instance_is_a',
                    'subject': subject,
                    'object': object,
                    'relation': relation,
                    'reason': reason,
                    'confidence': confidence
                })
                return False, reason
        
        # CRITICAL RULE 3: instance_of only for instance→class
        if relation == 'instance_of':
            if subj_name != 'instance' or obj_name == 'instance':
                reason = f"instance_of requires instance→class, got {subj_name}→{obj_name}"
                self.violations.append({
                    'type': 'invalid_instance_of',
                    'subject': subject,
                    'object': object,
                    'relation': relation,
                    'reason': reason,
                    'confidence': confidence
                })
                return False, reason
        
        return True, "Valid"
    
    def filter_relationships(self, relationships):
        """Filter a list of relationships, removing invalid ones"""
        valid = []
        
        for rel in relationships:
            # Extract info based on relationship type
            if hasattr(rel, 'subject'):
                subj = rel.subject
                obj = rel.object
                rel_type = getattr(rel, 'relation', 'unknown')
                conf = getattr(rel, 'weight', 1.0)
            elif isinstance(rel, dict):
                subj = rel.get('subject', '')
                obj = rel.get('object', '')
                rel_type = rel.get('relation', 'unknown')
                conf = rel.get('weight', 1.0)
            else:
                continue
            
            is_valid, reason = self.validate_relationship(subj, rel_type, obj, conf)
            
            if is_valid:
                valid.append(rel)
            else:
                print(f"  ❌ Rejected: {subj} → {obj} ({rel_type})")
                print(f"     Reason: {reason}")
        
        return valid
    
    def print_violations(self):
        """Print all detected violations"""
        if not self.violations:
            print("✅ No ontology violations found!")
            return
        
        print(f"\n⛔ FOUND {len(self.violations)} ONTOLOGY VIOLATIONS:")
        print("-" * 60)
        
        # Group by type
        by_type = {}
        for violation in self.violations:
            v_type = violation['type']
            if v_type not in by_type:
                by_type[v_type] = []
            by_type[v_type].append(violation)
        
        for v_type, violations in by_type.items():
            print(f"\n{self._get_violation_label(v_type)} ({len(violations)}):")
            for v in violations[:3]:  # Show first 3 of each type
                print(f"  {v['subject']} → {v['object']} ({v['relation']})")
                print(f"    Confidence: {v['confidence']:.2f}, Reason: {v['reason']}")
            
            if len(violations) > 3:
                print(f"  ... and {len(violations) - 3} more")
    
    def _get_violation_label(self, violation_type):
        """Get readable label for violation type"""
        labels = {
            'cross_domain_is_a': "CROSS-DOMAIN is_a",
            'instance_is_a': "INSTANCE USING is_a",
            'invalid_instance_of': "INVALID instance_of"
        }
        return labels.get(violation_type, violation_type.upper())

# Export
__all__ = ['OntologyGuard', 'OntologyDomain']
