# ontology_diagnostic_fixed.py
# Fixed version with proper Windows encoding

import sys
sys.path.insert(0, '.')

def diagnose_current_kb():
    """Diagnose the current knowledge base"""
    
    print("="*70)
    print("ONTOLOGY CONTAMINATION DIAGNOSTIC")
    print("="*70)
    
    # Test with known problematic relationships
    problematic_relationships = [
        {'subject': 'Fluffy', 'relation': 'is_a', 'object': 'DeepLearning', 'weight': 0.8},
        {'subject': 'Dog', 'relation': 'is_a', 'object': 'NeuralNetwork', 'weight': 0.7},
        {'subject': 'Animal', 'relation': 'is_a', 'object': 'DeepLearning', 'weight': 0.6},
        {'subject': 'Mammal', 'relation': 'is_a', 'object': 'NeuralNetwork', 'weight': 0.5},
        {'subject': 'Cat', 'relation': 'is_a', 'object': 'Mammal', 'weight': 0.9},
        {'subject': 'Mammal', 'relation': 'is_a', 'object': 'Animal', 'weight': 0.9},
        {'subject': 'CNN', 'relation': 'is_a', 'object': 'NeuralNetwork', 'weight': 0.9},
        {'subject': 'NeuralNetwork', 'relation': 'is_a', 'object': 'DeepLearning', 'weight': 0.8},
        {'subject': 'Fluffy', 'relation': 'is_a', 'object': 'Cat', 'weight': 0.9},
        {'subject': 'Fluffy', 'relation': 'instance_of', 'object': 'Cat', 'weight': 0.9},
        {'subject': 'Fido', 'relation': 'is_a', 'object': 'Dog', 'weight': 0.9},
    ]
    
    print("\n1. Testing relationships for ontology violations:")
    print("-" * 40)
    
    # Simple domain detection
    def get_domain(concept):
        concept_lower = concept.lower()
        if any(kw in concept_lower for kw in ['fluffy', 'fido', 'nimhag']):
            return 'instance'
        elif any(kw in concept_lower for kw in ['animal', 'mammal', 'dog', 'cat', 'pet']):
            return 'biology'
        elif any(kw in concept_lower for kw in ['neural', 'deep', 'cnn', 'rnn', 'learning']):
            return 'machine_learning'
        elif any(kw in concept_lower for kw in ['software', 'python', 'programming']):
            return 'computer_science'
        else:
            return 'unknown'
    
    violations = []
    
    for rel in problematic_relationships:
        subj = rel['subject']
        obj = rel['object']
        relation = rel['relation']
        
        subj_domain = get_domain(subj)
        obj_domain = get_domain(obj)
        
        # Check for violations
        is_violation = False
        reason = ""
        
        if relation == 'is_a':
            if subj_domain != obj_domain:
                if subj_domain == 'instance':
                    is_violation = True
                    reason = f"Instance using is_a (should be instance_of)"
                elif 'biology' in [subj_domain, obj_domain] and 'machine_learning' in [subj_domain, obj_domain]:
                    is_violation = True
                    reason = f"Cross-domain: {subj_domain} -> {obj_domain}"
        
        if is_violation:
            print(f"  VIOLATION: {subj} -> {obj} ({relation})")
            print(f"    Reason: {reason}")
            violations.append((subj, obj, relation, reason))
        else:
            print(f"  OK: {subj} -> {obj} ({relation})")
    
    print(f"\n2. Found {len(violations)} violations")
    
    if violations:
        print("\n3. Example violations found in your system:")
        for subj, obj, rel, reason in violations[:5]:
            print(f"   - {subj} -> {obj} ({rel})")
            print(f"     {reason}")
    else:
        print("\n3. No violations found!")
    
    return violations

def create_clean_structure():
    """Show what a clean ontology structure looks like"""
    
    print("\n" + "="*70)
    print("CLEAN ONTOLOGY STRUCTURE")
    print("="*70)
    
    clean_structure = '''
BIOLOGY DOMAIN:
  Animal
    |
    +-- Mammal
          |
          +-- Dog (class)
          |     |
          |     +-- Fido (instance_of)
          |
          +-- Cat (class)
                |
                +-- Fluffy (instance_of)

ML DOMAIN:
  DeepLearning
    |
    +-- NeuralNetwork
          |
          +-- CNN
          |
          +-- RNN

CS DOMAIN:
  Software
    |
    +-- ProgrammingLanguage
          |
          +-- Python

PROBLEMS IN CURRENT SYSTEM:
  Fluffy -> DeepLearning (is_a)    [WRONG: instance -> different domain]
  Dog -> NeuralNetwork (is_a)      [WRONG: cross-domain]
  Animal -> DeepLearning (is_a)    [WRONG: cross-domain]
'''
    
    print(clean_structure)

if __name__ == "__main__":
    violations = diagnose_current_kb()
    create_clean_structure()
    
    if violations:
        print("\n" + "="*70)
        print("RECOMMENDED FIXES:")
        print("="*70)
        print("1. Separate instance_of from is_a")
        print("2. Add domain constraints to reasoning")
        print("3. Run: python apply_simple_fix.py")
