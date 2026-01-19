# neurosql_complete_system.py
"""NeuroSQL Complete System with Inference Rules and Validation Constraints"""

print("="*70)
print("NEUROSQL COMPLETE SYSTEM")
print("="*70)

class OntologyGuard:
    """Validation constraints for neuroscience ontology"""
    
    def __init__(self):
        self.violations = []
        
        # Domain definitions
        self.domains = {
            'neurochemical': ['dopamine', 'serotonin', 'glutamate', 'gaba'],
            'brain_structure': ['hippocampus', 'prefrontal_cortex', 'amygdala'],
            'cognitive': ['memory', 'attention', 'emotion', 'learning'],
            'cellular': ['neuron', 'synapse', 'action_potential']
        }
        
        # VALIDATION CONSTRAINT 1: Domain compatibility rules
        self.domain_rules = {
            ('neurochemical', 'cognitive'): ['modulates', 'influences', 'affects'],
            ('brain_structure', 'cognitive'): ['supports', 'mediates', 'involved_in'],
            ('cellular', 'cellular'): ['connects_to', 'communicates_with']
        }
        
        # VALIDATION CONSTRAINT 2: Forbidden relationships
        self.forbidden = {
            'causes': "Causal claims require specific evidence",
            'contains': "Physical containment not applicable to abstract concepts",
            'is': "Avoid identity claims between different domains"
        }
    
    def validate_relationship(self, subject, relation, object):
        """Apply validation constraints"""
        
        # Constraint 1: Check forbidden relations
        if relation in self.forbidden:
            return False, self.forbidden[relation]
        
        # Constraint 2: Check domain compatibility
        subj_domain = self._get_domain(subject)
        obj_domain = self._get_domain(object)
        
        if subj_domain == 'unknown' or obj_domain == 'unknown':
            return True, "Unknown domain - allowing with caution"
        
        domain_pair = (subj_domain, obj_domain)
        if domain_pair in self.domain_rules:
            if relation not in self.domain_rules[domain_pair]:
                return False, f"'{relation}' not allowed between {subj_domain} and {obj_domain}"
        else:
            return False, f"No valid relations between {subj_domain} and {obj_domain}"
        
        return True, f"Valid {subj_domain}→{obj_domain} relation"
    
    def _get_domain(self, concept):
        """Determine which domain a concept belongs to"""
        concept_lower = concept.lower()
        for domain, concepts in self.domains.items():
            if concept_lower in concepts:
                return domain
        return 'unknown'

class InferenceEngine:
    """Inference rules for logical reasoning"""
    
    def __init__(self):
        self.inference_rules = []
        self._setup_rules()
    
    def _setup_rules(self):
        """Define inference rules"""
        
        # INFERENCE RULE 1: Transitive property (is_a hierarchy)
        self.inference_rules.append({
            'name': 'transitive_is_a',
            'pattern': [
                {'subject': '?A', 'relation': 'is_a', 'object': '?B'},
                {'subject': '?B', 'relation': 'is_a', 'object': '?C'}
            ],
            'inference': {'subject': '?A', 'relation': 'is_a', 'object': '?C'},
            'confidence': 0.9  # High confidence for transitive relationships
        })
        
        # INFERENCE RULE 2: Property inheritance
        self.inference_rules.append({
            'name': 'property_inheritance',
            'pattern': [
                {'subject': '?A', 'relation': 'is_a', 'object': '?B'},
                {'subject': '?B', 'relation': 'has_property', 'object': '?P'}
            ],
            'inference': {'subject': '?A', 'relation': 'has_property', 'object': '?P'},
            'confidence': 0.8  # Properties may not always be inherited
        })
        
        # INFERENCE RULE 3: Neuroscience-specific: Modulation chain
        self.inference_rules.append({
            'name': 'modulation_chain',
            'pattern': [
                {'subject': '?Neurotransmitter', 'relation': 'modulates', 'object': '?Region'},
                {'subject': '?Region', 'relation': 'supports', 'object': '?Function'}
            ],
            'inference': {'subject': '?Neurotransmitter', 'relation': 'influences', 'object': '?Function'},
            'confidence': 0.7  # Indirect influence
        })
    
    def apply_rules(self, knowledge_base):
        """Apply inference rules to generate new knowledge"""
        new_facts = []
        
        for rule in self.inference_rules:
            print(f"\nApplying rule: {rule['name']}")
            
            # For simplicity, we'll implement a basic pattern matcher
            # In a real system, you'd use a proper unification algorithm
            if rule['name'] == 'transitive_is_a':
                new_facts.extend(self._apply_transitive_rule(knowledge_base))
            elif rule['name'] == 'property_inheritance':
                new_facts.extend(self._apply_inheritance_rule(knowledge_base))
            elif rule['name'] == 'modulation_chain':
                new_facts.extend(self._apply_modulation_rule(knowledge_base))
        
        return new_facts
    
    def _apply_transitive_rule(self, kb):
        """Apply transitive is_a rule"""
        new_facts = []
        
        # Find all is_a relationships
        is_a_facts = [f for f in kb if f['relation'] == 'is_a']
        
        # Simple transitive closure
        for fact1 in is_a_facts:
            for fact2 in is_a_facts:
                if fact1['object'] == fact2['subject']:
                    new_fact = {
                        'subject': fact1['subject'],
                        'relation': 'is_a',
                        'object': fact2['object'],
                        'source': 'inference',
                        'confidence': 0.9
                    }
                    if new_fact not in kb and new_fact not in new_facts:
                        new_facts.append(new_fact)
                        print(f"  Inferred: {new_fact['subject']} → {new_fact['object']} (is_a)")
        
        return new_facts
    
    def _apply_inheritance_rule(self, kb):
        """Apply property inheritance rule"""
        new_facts = []
        
        is_a_facts = [f for f in kb if f['relation'] == 'is_a']
        property_facts = [f for f in kb if f['relation'] == 'has_property']
        
        for is_a in is_a_facts:
            for prop in property_facts:
                if is_a['object'] == prop['subject']:
                    new_fact = {
                        'subject': is_a['subject'],
                        'relation': 'has_property',
                        'object': prop['object'],
                        'source': 'inference',
                        'confidence': 0.8
                    }
                    if new_fact not in kb and new_fact not in new_facts:
                        new_facts.append(new_fact)
                        print(f"  Inferred: {new_fact['subject']} has_property {new_fact['object']}")
        
        return new_facts
    
    def _apply_modulation_chain(self, kb):
        """Apply neuroscience modulation chain rule"""
        new_facts = []
        
        modulate_facts = [f for f in kb if f['relation'] == 'modulates']
        supports_facts = [f for f in kb if f['relation'] == 'supports']
        
        for mod in modulate_facts:
            for sup in supports_facts:
                if mod['object'] == sup['subject']:
                    new_fact = {
                        'subject': mod['subject'],
                        'relation': 'influences',
                        'object': sup['object'],
                        'source': 'inference',
                        'confidence': 0.7
                    }
                    if new_fact not in kb and new_fact not in new_facts:
                        new_facts.append(new_fact)
                        print(f"  Inferred: {new_fact['subject']} influences {new_fact['object']}")
        
        return new_facts

class NeuroSQLComplete:
    """Complete NeuroSQL system with both features"""
    
    def __init__(self):
        self.knowledge_base = []
        self.ontology_guard = OntologyGuard()
        self.inference_engine = InferenceEngine()
        print("✓ System initialized with ontology guard and inference engine")
    
    def add_fact(self, subject, relation, object, source="user", confidence=1.0):
        """Add a fact with validation and inference"""
        
        print(f"\n[Adding] {subject} → {object} ({relation})")
        
        # Step 1: Apply validation constraints
        is_valid, reason = self.ontology_guard.validate_relationship(subject, relation, object)
        if not is_valid:
            print(f"❌ VALIDATION FAILED: {reason}")
            return False
        
        # Step 2: Add to knowledge base
        fact = {
            'subject': subject,
            'relation': relation,
            'object': object,
            'source': source,
            'confidence': confidence
        }
        
        # Check for duplicates
        if fact not in self.knowledge_base:
            self.knowledge_base.append(fact)
            print(f"✓ Added to knowledge base")
            
            # Step 3: Apply inference rules
            self._run_inferences()
            return True
        else:
            print(f"⚠ Fact already exists")
            return False
    
    def _run_inferences(self):
        """Run inference engine on current knowledge"""
        new_facts = self.inference_engine.apply_rules(self.knowledge_base)
        
        for fact in new_facts:
            # Validate inferences too!
            is_valid, reason = self.ontology_guard.validate_relationship(
                fact['subject'], fact['relation'], fact['object']
            )
            
            if is_valid and fact not in self.knowledge_base:
                self.knowledge_base.append(fact)
                print(f"✓ Added inference: {fact['subject']} → {fact['object']} ({fact['relation']})")
            elif not is_valid:
                print(f"❌ Inference rejected: {reason}")
    
    def query(self, pattern):
        """Query the knowledge base"""
        results = []
        for fact in self.knowledge_base:
            match = True
            for key, value in pattern.items():
                if fact.get(key) != value:
                    match = False
                    break
            if match:
                results.append(fact)
        return results
    
    def print_kb(self):
        """Print current knowledge base"""
        print(f"\n{'='*70}")
        print(f"KNOWLEDGE BASE ({len(self.knowledge_base)} facts)")
        print(f"{'='*70}")
        
        for i, fact in enumerate(self.knowledge_base, 1):
            source = fact['source']
            conf = fact['confidence']
            print(f"{i:2}. {fact['subject']:20} → {fact['object']:20} ({fact['relation']:12}) [{source:8}] conf:{conf:.1f}")

# DEMONSTRATION
neurosql = NeuroSQLComplete()

print("\n" + "="*70)
print("DEMONSTRATING VALIDATION CONSTRAINTS")
print("="*70)

print("\n--- Adding valid neuroscience facts ---")
neurosql.add_fact("dopamine", "is_a", "neurotransmitter")
neurosql.add_fact("neurotransmitter", "has_property", "chemical_messenger")
neurosql.add_fact("hippocampus", "is_a", "brain_structure")
neurosql.add_fact("brain_structure", "has_property", "neural_tissue")
neurosql.add_fact("dopamine", "modulates", "reward_circuit")
neurosql.add_fact("reward_circuit", "supports", "motivation")

print("\n--- Testing validation constraints ---")
neurosql.add_fact("dopamine", "causes", "happiness")  # Should fail
neurosql.add_fact("neuron", "contains", "consciousness")  # Should fail
neurosql.add_fact("hippocampus", "supports", "memory")  # Should succeed

print("\n" + "="*70)
print("DEMONSTRATING INFERENCE RULES")
print("="*70)

# Show inferences that were automatically generated
neurosql.print_kb()

print("\n" + "="*70)
print("QUERY EXAMPLES")
print("="*70)

# Query examples
print("\nAll facts about dopamine:")
for fact in neurosql.query({'subject': 'dopamine'}):
    print(f"  {fact['relation']} {fact['object']}")

print("\nAll inferred facts:")
for fact in neurosql.query({'source': 'inference'}):
    print(f"  {fact['subject']} → {fact['object']} ({fact['relation']})")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total facts in KB: {len(neurosql.knowledge_base)}")
print(f"Validation constraints prevent nonsense relationships")
print(f"Inference rules automatically derive new knowledge")
print("="*70)
