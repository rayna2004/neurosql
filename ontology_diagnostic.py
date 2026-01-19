# ontology_diagnostic.py
"""
Diagnose ontology contamination in NeuroSQL
"""

import sys
sys.path.insert(0, '.')

from ontology_guard import OntologyGuard

def diagnose_current_kb():
    """Diagnose the current knowledge base"""
    
    print("="*70)
    print("ONTOLOGY CONTAMINATION DIAGNOSTIC")
    print("="*70)
    
    guard = OntologyGuard()
    
    # Test with known problematic relationships from the demo
    problematic_relationships = [
        # These are what we saw in the visualization
        {'subject': 'Fluffy', 'relation': 'is_a', 'object': 'DeepLearning', 'weight': 0.8},
        {'subject': 'Dog', 'relation': 'is_a', 'object': 'NeuralNetwork', 'weight': 0.7},
        {'subject': 'Animal', 'relation': 'is_a', 'object': 'DeepLearning', 'weight': 0.6},
        {'subject': 'Mammal', 'relation': 'is_a', 'object': 'NeuralNetwork', 'weight': 0.5},
        
        # These should be valid
        {'subject': 'Cat', 'relation': 'is_a', 'object': 'Mammal', 'weight': 0.9},
        {'subject': 'Mammal', 'relation': 'is_a', 'object': 'Animal', 'weight': 0.9},
        {'subject': 'CNN', 'relation': 'is_a', 'object': 'NeuralNetwork', 'weight': 0.9},
        {'subject': 'NeuralNetwork', 'relation': 'is_a', 'object': 'DeepLearning', 'weight': 0.8},
        
        # Instance relationships (should be instance_of, not is_a)
        {'subject': 'Fluffy', 'relation': 'is_a', 'object': 'Cat', 'weight': 0.9},  # WRONG
        {'subject': 'Fluffy', 'relation': 'instance_of', 'object': 'Cat', 'weight': 0.9},  # CORRECT
        {'subject': 'Fido', 'relation': 'is_a', 'object': 'Dog', 'weight': 0.9},  # WRONG
    ]
    
    print("\n1. Testing individual relationships:")
    print("-" * 40)
    
    for rel in problematic_relationships:
        is_valid, reason = guard.validate_relationship(
            rel['subject'],
            rel['relation'],
            rel['object'],
            rel['weight']
        )
        
        status = "✅" if is_valid else "❌"
        print(f"{status} {rel['subject']} → {rel['object']} ({rel['relation']})")
        if not is_valid:
            print(f"   Reason: {reason}")
    
    # Test filtering
    print("\n2. Testing bulk filtering:")
    print("-" * 40)
    
    filtered = guard.filter_relationships(problematic_relationships)
    print(f"Original: {len(problematic_relationships)} relationships")
    print(f"After filtering: {len(filtered)} relationships")
    
    # Show violations
    print("\n3. Violation report:")
    print("-" * 40)
    guard.print_violations()
    
    # Show domain classification
    print("\n4. Domain classification:")
    print("-" * 40)
    
    test_concepts = ['Fluffy', 'Dog', 'Cat', 'Mammal', 'NeuralNetwork', 
                     'DeepLearning', 'Software', 'Python', 'Concept']
    
    for concept in test_concepts:
        domain = guard.get_domain(concept)
        domain_name = domain['name'] if isinstance(domain, dict) else domain
        print(f"  {concept:20} → {domain_name}")
    
    return guard.violations

def create_clean_structure():
    """Show what a clean ontology structure looks like"""
    
    print("\n" + "="*70)
    print("CLEAN ONTOLOGY STRUCTURE")
    print("="*70)
    
    clean_structure = '''
BIOLOGY DOMAIN:
  LivingBeing (abstract)
    │
    └── Animal (class)
          │
          └── Mammal (class)
                │
                ├── Dog (class)
                │     └── Fido (instance_of)  ✅
                │
                └── Cat (class)
                      └── Fluffy (instance_of)  ✅

ML DOMAIN:
  Algorithm (abstract)
    │
    └── MachineLearning (class)
          │
          └── NeuralNetwork (class)
                │
                ├── CNN (class)
                └── RNN (class)

CS DOMAIN:
  Concept (abstract)
    │
    └── Software (class)
          │
          └── ProgrammingLanguage (class)
                │
                └── Python (class)

VALID CROSS-DOMAIN (weak, non-inheritable):
  Software → Concept (is_a)           ✅ same abstract domain
  Animal → LivingBeing (is_a)         ✅ same abstract domain
  Python → Software (is_a)            ✅ same concrete domain
  
  Software → Concept (related_to)     ✅ weak cross-domain
  Animal → Concept (related_to)       ✅ weak cross-domain

INVALID (current system produces):
  Fluffy → DeepLearning (is_a)        ❌ instance→different domain
  Dog → NeuralNetwork (is_a)          ❌ cross-concrete-domain
  Animal → DeepLearning (is_a)        ❌ cross-concrete-domain
'''
    
    print(clean_structure)

if __name__ == "__main__":
    violations = diagnose_current_kb()
    create_clean_structure()
    
    if violations:
        print("\n" + "="*70)
        print("RECOMMENDED FIXES:")
        print("="*70)
        print("1. Run: .\apply_ontology_fix.ps1")
        print("2. Check neurosql_advanced.py for reasoning constraints")
        print("3. Separate instance_of from is_a relationships")
        print("4. Add domain validation before adding relationships")

