print("NEUROSQL ONTOLOGY PROTECTION DEMO")
print("=" * 50)

# Simple ontology guard implementation
class SimpleOntologyGuard:
    def __init__(self):
        self.valid_domains = {
            'neurochemical': ['dopamine', 'serotonin', 'glutamate', 'gaba'],
            'brain_region': ['prefrontal_cortex', 'hippocampus', 'amygdala'],
            'cognitive': ['memory', 'attention', 'emotion'],
            'cellular': ['neuron', 'synapse', 'action_potential']
        }
        
        self.allowed_relations = {
            ('neurochemical', 'cellular'): ['modulates', 'affects'],
            ('brain_region', 'cognitive'): ['supports', 'involved_in'],
            ('cellular', 'cellular'): ['connects_to', 'communicates_with']
        }
    
    def validate_relationship(self, subject, object, relation):
        # Find domains
        subj_domain = next((d for d, items in self.valid_domains.items() if subject in items), None)
        obj_domain = next((d for d, items in self.valid_domains.items() if object in items), None)
        
        if not subj_domain or not obj_domain:
            return False, f"Unknown domain: {subject}({subj_domain}) → {object}({obj_domain})"
        
        allowed = self.allowed_relations.get((subj_domain, obj_domain), [])
        if relation in allowed:
            return True, f"Valid: {subj_domain} → {obj_domain}"
        else:
            return False, f"Invalid relation between {subj_domain} and {obj_domain}"

# Test it
guard = SimpleOntologyGuard()

print("\\nTesting valid relationships:")
tests = [
    ("dopamine", "neuron", "modulates"),
    ("hippocampus", "memory", "supports"),
    ("neuron", "neuron", "connects_to")
]

for s, o, r in tests:
    valid, reason = guard.validate_relationship(s, o, r)
    status = "✓" if valid else "✗"
    print(f"{status} {s} → {o} ({r}): {reason}")

print("\\nTesting invalid relationships:")
invalid_tests = [
    ("dopamine", "happiness", "causes"),
    ("memory", "neuron", "stored_in"),
    ("prefrontal_cortex", "dopamine", "produces")
]

for s, o, r in invalid_tests:
    valid, reason = guard.validate_relationship(s, o, r)
    status = "✓" if valid else "✗"
    print(f"{status} {s} → {o} ({r}): {reason}")

print("\\n" + "=" * 50)
print("Ontology protection is working!")
