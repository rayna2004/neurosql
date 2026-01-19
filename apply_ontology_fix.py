# apply_ontology_fix.py
"""
Apply ontology fixes to NeuroSQL
"""

import sys
import os
sys.path.insert(0, '.')

from ontology_guard import OntologyGuard

def patch_neurosql_advanced():
    """Patch neurosql_advanced.py with ontology constraints"""
    
    print("="*70)
    print("PATCHING NEUROSQL_ADVANCED.PY")
    print("="*70)
    
    if not os.path.exists('neurosql_advanced.py'):
        print("❌ neurosql_advanced.py not found!")
        return False
    
    # Read the file
    with open('neurosql_advanced.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if 'OntologyGuard' in content:
        print("✅ neurosql_advanced.py already has OntologyGuard")
        return True
    
    # Find the demo method
    demo_start = content.find('def demo(self):')
    if demo_start == -1:
        print("❌ Could not find demo() method")
        return False
    
    # Find where to insert imports
    import_section = content.find('from reasoning_engine import')
    if import_section == -1:
        import_section = content.find('from query_engine import')
    
    if import_section == -1:
        print("❌ Could not find import section")
        return False
    
    # Add import
    new_import = '\nfrom ontology_guard import OntologyGuard\n'
    content = content[:import_section] + new_import + content[import_section:]
    
    # Find the reasoning section
    reasoning_section = content.find('3. Applying reasoning:', demo_start)
    if reasoning_section == -1:
        print("❌ Could not find reasoning section")
        return False
    
    # Create the patch for transitive closure
    patch = '''

        # ONTOLOGY FIX: Add constraint checking to transitive closure
        print("\\n   a) Constrained Transitive Closure:")
        guard = OntologyGuard()
        valid_inferences = []
        
        # Get existing is_a relationships
        is_a_rels = [r for r in self.neurosql.relationships 
                    if hasattr(r, 'relation') and r.relation == 'is_a']
        
        # Apply transitive closure with constraints
        for rel1 in is_a_rels:
            for rel2 in is_a_rels:
                if rel1.object == rel2.subject:
                    # Candidate inference: rel1.subject → rel2.object
                    is_valid, reason = guard.validate_relationship(
                        rel1.subject, 'is_a', rel2.object,
                        min(rel1.weight, rel2.weight) * 0.9
                    )
                    
                    if is_valid:
                        # Add inferred relationship
                        from neurosql_core import WeightedRelationship, RelationshipType
                        inferred = WeightedRelationship(
                            rel1.subject, rel2.object, 
                            RelationshipType.IS_A,
                            min(rel1.weight, rel2.weight) * 0.9
                        )
                        self.neurosql.add_weighted_relationship(inferred)
                        valid_inferences.append(f"{rel1.subject} → {rel2.object}")
                    else:
                        print(f"     ❌ Rejected: {rel1.subject} → {rel2.object}")
                        print(f"        Reason: {reason}")
        
        print(f"   Inferred {len(valid_inferences)} VALID relationships")
        if valid_inferences:
            print(f"   Steps: {', '.join(valid_inferences[:5])}" + 
                  ("..." if len(valid_inferences) > 5 else ""))
        
        # Skip the old transitive closure
        print("\\n   b) Property Inheritance:")
        # ... rest of original code continues ...
'''
    
    # Find where to insert the patch (after "a) Transitive Closure:")
    transitive_start = content.find('a) Transitive Closure:', reasoning_section)
    if transitive_start == -1:
        print("❌ Could not find transitive closure section")
        return False
    
    # Find the end of the transitive closure section
    next_section = content.find('b) Property Inheritance:', transitive_start)
    if next_section == -1:
        next_section = content.find('c) Default Reasoning:', transitive_start)
    
    if next_section == -1:
        print("❌ Could not find next section")
        return False
    
    # Replace the entire transitive closure section
    content = content[:transitive_start] + patch + content[next_section:]
    
    # Create backup
    import shutil
    shutil.copy2('neurosql_advanced.py', 'neurosql_advanced.py.backup')
    print("✅ Backup created: neurosql_advanced.py.backup")
    
    # Write the patched file
    with open('neurosql_advanced.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Successfully patched neurosql_advanced.py")
    print("   - Added OntologyGuard import")
    print("   - Replaced transitive closure with constrained version")
    print("   - Backup saved")
    
    return True

def create_clean_demo():
    """Create a clean demo file with proper ontology"""
    
    print("\n" + "="*70)
    print("CREATING CLEAN DEMO: neurosql_clean.py")
    print("="*70)
    
    clean_demo = '''# neurosql_clean.py
"""
Clean NeuroSQL demo with proper ontology separation
"""

from neurosql_core import NeuroSQL, Concept, WeightedRelationship, RelationshipType
from semantics import SemanticsEngine
from query_engine import NeuroSQLWithQuery
from reasoning_engine import ReasoningEngine, ReasoningOperator
from ontology_guard import OntologyGuard

class NeuroSQLClean(NeuroSQLWithQuery):
    def __init__(self, name: str = "neurosql_clean"):
        super().__init__(name)
        self.reasoning_engine = ReasoningEngine(self.neurosql)
        self.semantics = SemanticsEngine()
        self.ontology_guard = OntologyGuard()

    def demo(self):
        print("=" * 60)
        print("NEUROSQL CLEAN DEMO - PROPER ONTOLOGY")
        print("=" * 60)

        # Create CLEAN knowledge base with proper ontology
        concepts = [
            # BIOLOGY DOMAIN
            Concept("LivingBeing", {"ontology": "abstract", "type": "class"}, 1, "abstract"),
            Concept("Animal", {"ontology": "biology", "type": "class"}, 2, "biology"),
            Concept("Mammal", {"ontology": "biology", "type": "class"}, 3, "biology"),
            Concept("Dog", {"ontology": "biology", "type": "class"}, 4, "biology"),
            Concept("Cat", {"ontology": "biology", "type": "class"}, 4, "biology"),
            Concept("Pet", {"ontology": "biology", "type": "role"}, 3, "biology"),
            
            # INSTANCES (proper nouns)
            Concept("Fluffy", {"ontology": "instance", "type": "instance"}, 1, "instance"),
            Concept("Fido", {"ontology": "instance", "type": "instance"}, 1, "instance"),
            
            # ML DOMAIN
            Concept("Algorithm", {"ontology": "abstract", "type": "class"}, 1, "abstract"),
            Concept("MachineLearning", {"ontology": "ml", "type": "class"}, 2, "ml"),
            Concept("NeuralNetwork", {"ontology": "ml", "type": "class"}, 3, "ml"),
            Concept("CNN", {"ontology": "ml", "type": "class"}, 4, "ml"),
            Concept("RNN", {"ontology": "ml", "type": "class"}, 4, "ml"),
            Concept("DeepLearning", {"ontology": "ml", "type": "class"}, 3, "ml"),
            
            # CS DOMAIN
            Concept("Concept", {"ontology": "abstract", "type": "class"}, 1, "abstract"),
            Concept("Software", {"ontology": "cs", "type": "class"}, 2, "cs"),
            Concept("ProgrammingLanguage", {"ontology": "cs", "type": "class"}, 3, "cs"),
            Concept("Python", {"ontology": "cs", "type": "class"}, 4, "cs"),
        ]

        for concept in concepts:
            self.neurosql.add_concept(concept)

        # CLEAN relationships with proper types
        relationships = [
            # BIOLOGY: class → class (is_a)
            WeightedRelationship("Cat", "Mammal", RelationshipType.IS_A, 0.98),
            WeightedRelationship("Dog", "Mammal", RelationshipType.IS_A, 0.98),
            WeightedRelationship("Mammal", "Animal", RelationshipType.IS_A, 0.97),
            WeightedRelationship("Animal", "LivingBeing", RelationshipType.IS_A, 0.95),
            WeightedRelationship("Pet", "Animal", RelationshipType.IS_A, 0.85),
            
            # INSTANCES: instance → class (instance_of)
            WeightedRelationship("Fluffy", "Cat", "instance_of", 1.0),
            WeightedRelationship("Fido", "Dog", "instance_of", 1.0),
            
            # ML: class → class (is_a)
            WeightedRelationship("CNN", "NeuralNetwork", RelationshipType.IS_A, 0.95),
            WeightedRelationship("RNN", "NeuralNetwork", RelationshipType.IS_A, 0.95),
            WeightedRelationship("NeuralNetwork", "MachineLearning", RelationshipType.IS_A, 0.90),
            WeightedRelationship("MachineLearning", "Algorithm", RelationshipType.IS_A, 0.85),
            WeightedRelationship("DeepLearning", "MachineLearning", RelationshipType.IS_A, 0.88),
            
            # CS: class → class (is_a)
            WeightedRelationship("Python", "ProgrammingLanguage", RelationshipType.IS_A, 0.99),
            WeightedRelationship("ProgrammingLanguage", "Software", RelationshipType.IS_A, 0.85),
            WeightedRelationship("Software", "Concept", RelationshipType.IS_A, 0.80),
            
            # CROSS-DOMAIN: weak relationships only
            WeightedRelationship("Software", "Concept", "related_to", 0.3),
            WeightedRelationship("Animal", "LivingBeing", "related_to", 0.3),
        ]

        for rel in relationships:
            # Validate before adding
            if hasattr(rel, 'relation'):
                rel_type = rel.relation
            else:
                rel_type = 'unknown'
            
            is_valid, reason = self.ontology_guard.validate_relationship(
                rel.subject, rel_type, rel.object, rel.weight
            )
            
            if is_valid:
                self.neurosql.add_weighted_relationship(rel)
            else:
                print(f"  ⚠ Skipping invalid: {rel.subject} → {rel.object} ({rel_type})")
                print(f"     Reason: {reason}")

        print(f"\\n1. Created CLEAN knowledge base:")
        print(f"   Concepts: {len(self.neurosql.concepts)}")
        print(f"   Relationships: {len(self.neurosql.relationships)}")

        print("\\n2. Running queries:")

        # Query examples
        queries = [
            'GET CONCEPTS WHERE ontology = "biology"',
            'GET CONCEPTS WHERE ontology = "ml"',
            'FIND PATH FROM "Fluffy" TO "LivingBeing"',
            'FIND PATH FROM "CNN" TO "Algorithm"',
        ]

        for query in queries:
            print(f"\\n   Query: {query}")
            result = self.execute_query(query)
            print(f"   Results: {len(result) if result else 0}")

        print("\\n3. Applying CONSTRAINED reasoning:")

        # Run transitive closure with constraints
        print("\\n   a) Constrained Transitive Closure:")
        
        # Get existing is_a relationships
        is_a_rels = [r for r in self.neurosql.relationships 
                    if hasattr(r, 'relation') and r.relation == 'is_a']
        
        valid_inferences = []
        
        for rel1 in is_a_rels:
            for rel2 in is_a_rels:
                if rel1.object == rel2.subject:
                    is_valid, reason = self.ontology_guard.validate_relationship(
                        rel1.subject, 'is_a', rel2.object,
                        min(rel1.weight, rel2.weight) * 0.9
                    )
                    
                    if is_valid:
                        inferred = WeightedRelationship(
                            rel1.subject, rel2.object, 
                            RelationshipType.IS_A,
                            min(rel1.weight, rel2.weight) * 0.9
                        )
                        self.neurosql.add_weighted_relationship(inferred)
                        valid_inferences.append(f"{rel1.subject} → {rel2.object}")
                    else:
                        print(f"     ❌ Rejected: {rel1.subject} → {rel2.object}")
                        print(f"        Reason: {reason}")
        
        print(f"   Inferred {len(valid_inferences)} VALID relationships")
        if valid_inferences:
            print(f"   Example: {valid_inferences[0] if valid_inferences else 'none'}")

        print("\\n4. Final statistics:")
        print(f"   Concepts: {len(self.neurosql.concepts)}")
        print(f"   Relationships: {len(self.neurosql.relationships)}")
        
        # Count by type
        is_a_count = sum(1 for r in self.neurosql.relationships 
                        if hasattr(r, 'relation') and r.relation == 'is_a')
        instance_of_count = sum(1 for r in self.neurosql.relationships 
                               if hasattr(r, 'relation') and r.relation == 'instance_of')
        
        print(f"   is_a relationships: {is_a_count}")
        print(f"   instance_of relationships: {instance_of_count}")

        print("\\n" + "=" * 60)
        print("CLEAN DEMO COMPLETE - NO ONTOLOGY CONTAMINATION!")
        print("=" * 60)

if __name__ == "__main__":
    ns = NeuroSQLClean()
    ns.demo()
'''
    
    with open('neurosql_clean.py', 'w', encoding='utf-8') as f:
        f.write(clean_demo)
    
    print("✅ Created: neurosql_clean.py")
    print("   - Proper ontology separation")
    print("   - Instance vs class distinction")
    print("   - Constrained reasoning")
    
    return True

def main():
    """Main function"""
    
    # Run diagnostic first
    print("Running ontology diagnostic...")
    import ontology_diagnostic
    ontology_diagnostic.diagnose_current_kb()
    
    # Ask user
    print("\n" + "="*70)
    print("OPTIONS:")
    print("="*70)
    print("1. Patch existing neurosql_advanced.py")
    print("2. Create clean demo (recommended)")
    print("3. Both")
    print("4. Exit")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice in ['1', '3']:
            patch_neurosql_advanced()
        
        if choice in ['2', '3']:
            create_clean_demo()
            print("\n✅ Now run: python neurosql_clean.py")
        
        if choice == '4':
            print("Exiting...")
            return
        
        if choice not in ['1', '2', '3', '4']:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\nExiting...")
        return
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Run the clean demo: python neurosql_clean.py")
    print("2. Check that no cross-ontology inferences occur")
    print("3. Integrate OntologyGuard into your production code")
    print("4. Consider adding ontology learning from data")

if __name__ == "__main__":
    main()
