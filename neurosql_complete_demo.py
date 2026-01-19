# neurosql_ontology_demo.py
"""Complete NeuroSQL demo with ontology protection"""

print("="*70)
print("NEUROSQL WITH ONTOLOGY PROTECTION - COMPLETE DEMO")
print("="*70)

# Import ontology guard
try:
    from ontology_guard import OntologyGuard
    guard = OntologyGuard()
    print("✓ OntologyGuard loaded")
except:
    print("Creating simple ontology guard...")
    # Fallback simple implementation
    class SimpleGuard:
        def validate_relationship(self, subject, object, relation):
            # Simple rule-based validation
            nonsense_pairs = [
                ("dopamine", "happiness"),
                ("serotonin", "love"), 
                ("neuron", "idea"),
                ("synapse", "memory")
            ]
            
            for s, o in nonsense_pairs:
                if subject.lower() == s and object.lower() == o:
                    return False, f"Invalid cross-domain: {s} cannot directly relate to {o}"
            
            return True, "Valid relationship"

    guard = SimpleGuard()

# Simulate NeuroSQL system
class NeuroSQLSystem:
    def __init__(self):
        self.knowledge_base = []
        print("✓ NeuroSQL system initialized")
    
    def add_fact(self, subject, object, relation, source="user"):
        """Add a fact with ontology validation"""
        print(f"\nAttempting to add: {subject} → {object} ({relation})")
        
        # Check ontology
        is_valid, reason = guard.validate_relationship(subject, object, relation)
        
        if not is_valid:
            print(f"❌ REJECTED: {reason}")
            return False
        
        # Add to knowledge base
        fact = {
            'subject': subject,
            'object': object, 
            'relation': relation,
            'source': source
        }
        self.knowledge_base.append(fact)
        print(f"✓ Added to knowledge base")
        return True
    
    def query(self, question):
        """Answer questions based on knowledge"""
        print(f"\nQuery: {question}")
        
        # Simple keyword matching
        if "dopamine" in question.lower():
            return "Dopamine is a neurotransmitter involved in reward and motivation."
        elif "memory" in question.lower():
            return "Memory involves hippocampal formation and synaptic plasticity."
        else:
            return f"I have {len(self.knowledge_base)} facts in my knowledge base."

# Create and demonstrate the system
neurosql = NeuroSQLSystem()

print("\n" + "="*70)
print("DEMONSTRATING ONTOLOGY PROTECTION")
print("="*70)

# Valid neuroscience facts
print("\n--- VALID FACTS (will be accepted) ---")
neurosql.add_fact("dopamine", "reward_system", "modulates")
neurosql.add_fact("hippocampus", "spatial_memory", "supports")
neurosql.add_fact("prefrontal_cortex", "decision_making", "mediates")

print("\n--- INVALID FACTS (will be rejected) ---")
neurosql.add_fact("dopamine", "happiness", "causes")
neurosql.add_fact("neuron", "consciousness", "contains")
neurosql.add_fact("synapse", "idea", "stores")

print("\n" + "="*70)
print("QUERY DEMONSTRATION")
print("="*70)

# Try some queries
print(neurosql.query("What is dopamine?"))
print(neurosql.query("Tell me about memory"))
print(neurosql.query("What do you know?"))

print("\n" + "="*70)
print(f"Knowledge base contains {len(neurosql.knowledge_base)} valid facts")
print("="*70)
