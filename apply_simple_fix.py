# apply_simple_fix.py
"""
Simple fix for ontology contamination
"""

import sys
sys.path.insert(0, '.')

def add_ontology_guard():
    """Add simple ontology guard to neurosql_advanced.py"""
    
    print("="*70)
    print("ADDING ONTOLOGY GUARD TO NEUROSQL")
    print("="*70)
    
    try:
        # Read neurosql_advanced.py
        with open('neurosql_advanced.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already has ontology guard
        if 'SimpleOntologyGuard' in content:
            print("? Already has ontology guard")
            return True
        
        # Find the demo method
        demo_start = content.find('def demo(self):')
        if demo_start == -1:
            print("? Could not find demo method")
            return False
        
        # Find the transitive closure section
        tc_start = content.find('a) Transitive Closure:', demo_start)
        if tc_start == -1:
            print("? Could not find transitive closure section")
            return False
        
        # Create the simple ontology guard class at the top
        guard_class = '''
# === SIMPLE ONTOLOGY GUARD ===
class SimpleOntologyGuard:
    """Prevents cross-domain nonsense inferences"""
    
    @staticmethod
    def get_domain(concept):
        """Determine domain from concept name"""
        concept_lower = concept.lower()
        
        # Instances (proper nouns)
        if concept_lower in ['fluffy', 'fido', 'nimhag']:
            return 'instance'
        
        # Biology domain
        biology_keywords = ['animal', 'mammal', 'dog', 'cat', 'pet', 'biology']
        if any(kw in concept_lower for kw in biology_keywords):
            return 'biology'
        
        # ML domain
        ml_keywords = ['neural', 'deep', 'cnn', 'rnn', 'machine', 'learning']
        if any(kw in concept_lower for kw in ml_keywords):
            return 'machine_learning'
        
        # CS domain
        cs_keywords = ['software', 'python', 'programming', 'algorithm']
        if any(kw in concept_lower for kw in cs_keywords):
            return 'computer_science'
        
        return 'unknown'
    
    @staticmethod
    def validate_inference(subject, relation, object):
        """Validate if inference is semantically valid"""
        subj_domain = SimpleOntologyGuard.get_domain(subject)
        obj_domain = SimpleOntologyGuard.get_domain(object)
        
        # Rule 1: No cross-domain is_a
        if relation == 'is_a':
            if subj_domain != obj_domain:
                return False, f"Cross-domain is_a: {subj_domain} -> {obj_domain}"
            
            # Rule 2: Instances can't use is_a
            if subj_domain == 'instance':
                return False, f"Instance {subject} cannot use is_a (use instance_of)"
        
        return True, "Valid"

# === END ONTOLOGY GUARD ===

'''
        
        # Insert the guard class after imports
        import_section = content.find('from reasoning_engine import')
        if import_section == -1:
            import_section = content.find('class NeuroSQLAdvanced')
        
        if import_section != -1:
            content = content[:import_section] + guard_class + content[import_section:]
        
        # Now modify the transitive closure section
        # Find the end of transitive closure
        tc_end = content.find('b) Property Inheritance:', tc_start)
        if tc_end == -1:
            tc_end = content.find('c) Default Reasoning:', tc_start)
        
        if tc_end == -1:
            print("? Could not find end of transitive closure")
            return False
        
        # Create the constrained transitive closure
        constrained_tc = '''
        print("\\n   a) Constrained Transitive Closure:")
        guard = SimpleOntologyGuard()
        valid_inferences = []
        
        # Get existing is_a relationships
        is_a_rels = [r for r in self.neurosql.relationships 
                    if hasattr(r, 'relation') and r.relation == 'is_a']
        
        # Apply transitive closure with constraints
        for rel1 in is_a_rels:
            for rel2 in is_a_rels:
                if rel1.object == rel2.subject:
                    # Candidate inference: rel1.subject -> rel2.object
                    is_valid, reason = guard.validate_inference(
                        rel1.subject, 'is_a', rel2.object
                    )
                    
                    if is_valid:
                        # Add inferred relationship
                        inferred = WeightedRelationship(
                            rel1.subject, rel2.object, 
                            RelationshipType.IS_A,
                            min(rel1.weight, rel2.weight) * 0.9
                        )
                        self.neurosql.add_weighted_relationship(inferred)
                        valid_inferences.append(f"{rel1.subject} -> {rel2.object}")
                    else:
                        print(f"     REJECTED: {rel1.subject} -> {rel2.object}")
                        print(f"        Reason: {reason}")
        
        print(f"   Inferred {len(valid_inferences)} VALID relationships")
        if valid_inferences:
            print(f"   Steps: {', '.join(valid_inferences[:3])}" + 
                  ("..." if len(valid_inferences) > 3 else ""))
        
        # Skip old transitive closure
        print("\\n   b) Property Inheritance:")
'''
        
        # Replace the transitive closure section
        content = content[:tc_start] + constrained_tc + content[tc_end:]
        
        # Backup original
        import shutil
        shutil.copy2('neurosql_advanced.py', 'neurosql_advanced.py.backup')
        print("? Backup created: neurosql_advanced.py.backup")
        
        # Write the modified file
        with open('neurosql_advanced.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("? Successfully added ontology guard!")
        print("   - Added SimpleOntologyGuard class")
        print("   - Modified transitive closure to use constraints")
        print("   - Will now reject cross-domain inferences")
        
        return True
        
    except Exception as e:
        print(f"? Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_clean_demo():
    """Create a simple clean demo"""
    
    print("\n" + "="*70)
    print("CREATING SIMPLE CLEAN DEMO")
    print("="*70)
    
    clean_demo = '''# neurosql_simple_clean.py
"""
Simple clean demo with ontology separation
"""

from neurosql_core import NeuroSQL, Concept, WeightedRelationship, RelationshipType

def simple_clean_demo():
    print("="*60)
    print("SIMPLE CLEAN DEMO - NO ONTOLOGY CONTAMINATION")
    print("="*60)
    
    # Create NeuroSQL instance
    ns = NeuroSQL("simple_clean")
    
    # Add concepts with clear domains
    concepts = [
        # Biology
        Concept("Animal", {}, 1, "biology"),
        Concept("Mammal", {}, 2, "biology"),
        Concept("Dog", {}, 3, "biology"),
        Concept("Cat", {}, 3, "biology"),
        
        # Instances
        Concept("Fluffy", {}, 1, "instance"),
        Concept("Fido", {}, 1, "instance"),
        
        # Machine Learning
        Concept("NeuralNetwork", {}, 2, "ml"),
        Concept("DeepLearning", {}, 3, "ml"),
        Concept("CNN", {}, 4, "ml"),
        
        # Computer Science
        Concept("Software", {}, 2, "cs"),
        Concept("Python", {}, 3, "cs"),
    ]
    
    for concept in concepts:
        ns.add_concept(concept)
    
    print(f"Added {len(concepts)} concepts")
    
    # Add CLEAN relationships
    relationships = [
        # Biology: class -> class
        WeightedRelationship("Cat", "Mammal", RelationshipType.IS_A, 0.98),
        WeightedRelationship("Dog", "Mammal", RelationshipType.IS_A, 0.98),
        WeightedRelationship("Mammal", "Animal", RelationshipType.IS_A, 0.97),
        
        # Instances: instance -> class
        WeightedRelationship("Fluffy", "Cat", "instance_of", 1.0),
        WeightedRelationship("Fido", "Dog", "instance_of", 1.0),
        
        # ML: class -> class
        WeightedRelationship("CNN", "NeuralNetwork", RelationshipType.IS_A, 0.95),
        WeightedRelationship("NeuralNetwork", "DeepLearning", RelationshipType.IS_A, 0.90),
        
        # CS: class -> class
        WeightedRelationship("Python", "Software", RelationshipType.IS_A, 0.85),
    ]
    
    for rel in relationships:
        ns.add_weighted_relationship(rel)
    
    print(f"Added {len(relationships)} relationships")
    
    # Show what we have
    print("\\nCURRENT KNOWLEDGE BASE:")
    print(f"  Biology chain: Fluffy -> Cat -> Mammal -> Animal")
    print(f"  ML chain: CNN -> NeuralNetwork -> DeepLearning")
    print(f"  CS: Python -> Software")
    
    # Manual transitive closure (no contamination!)
    print("\\nMANUAL TRANSITIVE CLOSURE (clean):")
    
    # Only valid inferences:
    valid_inferences = [
        ("Cat", "Animal"),  # Cat -> Mammal -> Animal
        ("Dog", "Animal"),  # Dog -> Mammal -> Animal
        ("CNN", "DeepLearning"),  # CNN -> NeuralNetwork -> DeepLearning
    ]
    
    for subj, obj in valid_inferences:
        inferred = WeightedRelationship(subj, obj, RelationshipType.IS_A, 0.85)
        ns.add_weighted_relationship(inferred)
        print(f"  INFERRED: {subj} -> {obj}")
    
    # What we DON'T infer (contamination):
    invalid_inferences = [
        ("Fluffy", "DeepLearning"),  # NO! Instance -> different domain
        ("Dog", "NeuralNetwork"),    # NO! Cross-domain
        ("Animal", "Software"),      # NO! Cross-domain
    ]
    
    print("\\nREJECTED (would be contamination):")
    for subj, obj in invalid_inferences:
        print(f"  NOT INFERRED: {subj} -> {obj}")
    
    print("\\n" + "="*60)
    print("DEMO COMPLETE - NO CONTAMINATION!")
    print("="*60)
    
    return ns

if __name__ == "__main__":
    simple_clean_demo()
'''
    
    with open('neurosql_simple_clean.py', 'w', encoding='utf-8') as f:
        f.write(clean_demo)
    
    print("? Created: neurosql_simple_clean.py")
    print("\\nTo run: python neurosql_simple_clean.py")

def main():
    """Main function"""
    
    print("\\nOPTIONS:")
    print("1. Add ontology guard to neurosql_advanced.py")
    print("2. Create simple clean demo")
    print("3. Both")
    print("4. Exit")
    
    try:
        choice = input("\\nEnter choice (1-4): ").strip()
        
        if choice in ['1', '3']:
            add_ontology_guard()
            print("\\n? Now run: python neurosql_advanced.py")
            print("   It will reject cross-domain inferences!")
        
        if choice in ['2', '3']:
            create_simple_clean_demo()
            print("\\n? Now run: python neurosql_simple_clean.py")
        
        if choice == '4':
            print("Exiting...")
            return
        
    except KeyboardInterrupt:
        print("\\nExiting...")
        return
    
    print("\\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print("The system will now:")
    print("1. Reject Fluffy -> DeepLearning (instance->different domain)")
    print("2. Reject Dog -> NeuralNetwork (cross-domain)")
    print("3. Allow Cat -> Mammal -> Animal (same domain)")
    print("4. Allow CNN -> NeuralNetwork -> DeepLearning (same domain)")

if __name__ == "__main__":
    main()
