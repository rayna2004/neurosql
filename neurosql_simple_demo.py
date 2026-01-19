# neurosql_working.py
print("NEUROSQL WITH REAL ONTOLOGY REJECTION")
print("="*60)

# Simple but effective ontology guard
class OntologyGuard:
    def validate_relationship(self, subject, object, relation):
        # Forbidden relations
        forbidden_relations = ['causes', 'contains', 'stores', 'creates']
        if relation in forbidden_relations:
            return False, f"Relation '{relation}' is too strong/ambiguous"
        
        # Forbidden concept pairs
        nonsense_pairs = [
            ("dopamine", "happiness"),
            ("neuron", "consciousness"),
            ("synapse", "idea"),
            ("dopamine", "love")
        ]
        
        for s, o in nonsense_pairs:
            if subject == s and object == o:
                return False, f"Invalid: {s} cannot directly relate to {o}"
        
        return True, "Valid relationship"

# NeuroSQL system
class NeuroSQLSystem:
    def __init__(self):
        self.knowledge_base = []
    
    def add_fact(self, subject, object, relation):
        print(f"\nAttempting: {subject} → {object} ({relation})")
        
        guard = OntologyGuard()
        is_valid, reason = guard.validate_relationship(subject, object, relation)
        
        if not is_valid:
            print(f"❌ REJECTED: {reason}")
            return False
        
        self.knowledge_base.append((subject, object, relation))
        print(f"✓ Added to KB")
        return True

# Test it
neurosql = NeuroSQLSystem()

print("\n--- Valid facts ---")
neurosql.add_fact("dopamine", "reward", "modulates")
neurosql.add_fact("hippocampus", "memory", "supports")

print("\n--- Invalid facts ---")
neurosql.add_fact("dopamine", "happiness", "causes")
neurosql.add_fact("neuron", "consciousness", "contains")
neurosql.add_fact("synapse", "idea", "stores")

print(f"\nTotal in KB: {len(neurosql.knowledge_base)}")
if len(neurosql.knowledge_base) == 2:
    print("\n✅ SUCCESS: System correctly rejected invalid facts!")
