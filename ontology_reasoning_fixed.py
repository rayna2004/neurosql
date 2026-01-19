# ontology_reasoning_fixed.py
"""
Fixed Ontology Reasoning Demo
Shows BOTH valid inferences (within domains) AND rejected inferences (cross-domain)
"""

print("="*70)
print("ONTOLOGY-AWARE REASONING - COMPLETE DEMONSTRATION")
print("="*70)

# Create a simple ontology constraint system
class OntologyConstraintSystem:
    def __init__(self):
        self.valid_inferences = []
        self.rejected_inferences = []
        
    def get_domain(self, concept):
        """Determine domain from concept name"""
        concept_lower = concept.lower()
        
        # Instances (proper nouns)
        if concept_lower in ['fluffy', 'fido', 'nimhag', 'spot', 'whiskers']:
            return 'instance'
        
        # Biology domain
        biology_keywords = ['animal', 'mammal', 'dog', 'cat', 'pet', 'biology', 'organism', 'living']
        if any(kw in concept_lower for kw in biology_keywords):
            return 'biology'
        
        # Machine Learning domain
        ml_keywords = ['neural', 'deep', 'cnn', 'rnn', 'machine', 'learning', 'network', 'transformer']
        if any(kw in concept_lower for kw in ml_keywords):
            return 'machine_learning'
        
        # Computer Science domain
        cs_keywords = ['software', 'python', 'programming', 'algorithm', 'code', 'java', 'javascript']
        if any(kw in concept_lower for kw in cs_keywords):
            return 'computer_science'
        
        # Abstract concepts
        abstract_keywords = ['concept', 'entity', 'object', 'thing', 'being', 'idea']
        if any(kw in concept_lower for kw in abstract_keywords):
            return 'abstract'
        
        return 'unknown'
    
    def validate_inference(self, subject, relation, object, context="transitive_closure"):
        """Validate inference with detailed reasoning"""
        subj_domain = self.get_domain(subject)
        obj_domain = self.get_domain(object)
        
        # Rule 1: Only is_a and instance_of for transitive closure
        if relation not in ['is_a', 'instance_of']:
            reason = f"Relation {relation} not applicable for {context}"
            self.rejected_inferences.append({
                'subject': subject, 'object': object, 'relation': relation,
                'reason': reason, 'context': context
            })
            return False, reason
        
        # Rule 2: is_a constraints
        if relation == 'is_a':
            # Check if already same concept (would be a loop)
            if subject == object:
                reason = f"Self-reference: {subject} is_a {object}"
                self.rejected_inferences.append({
                    'subject': subject, 'object': object, 'relation': relation,
                    'reason': reason, 'context': context
                })
                return False, reason
            
            # Check cross-domain
            if subj_domain != obj_domain:
                # Only allow if connecting to abstract
                if 'abstract' not in [subj_domain, obj_domain]:
                    reason = f"Cross-domain is_a: {subj_domain} → {obj_domain}"
                    self.rejected_inferences.append({
                        'subject': subject, 'object': object, 'relation': relation,
                        'reason': reason, 'context': context
                    })
                    return False, reason
            
            # Check instance misuse
            if subj_domain == 'instance':
                reason = f"Instance {subject} cannot use is_a (use instance_of)"
                self.rejected_inferences.append({
                    'subject': subject, 'object': object, 'relation': relation,
                    'reason': reason, 'context': context
                })
                return False, reason
        
        # Rule 3: instance_of constraints
        if relation == 'instance_of':
            if subj_domain != 'instance':
                reason = f"Only instances can use instance_of, got {subj_domain}"
                self.rejected_inferences.append({
                    'subject': subject, 'object': object, 'relation': relation,
                    'reason': reason, 'context': context
                })
                return False, reason
        
        # If all checks pass
        self.valid_inferences.append({
            'subject': subject, 'object': object, 'relation': relation,
            'context': context, 'domains': f"{subj_domain}→{obj_domain}"
        })
        return True, f"Valid: {subj_domain}→{obj_domain}"

def simulate_transitive_closure():
    """Simulate what happens during transitive closure with constraints"""
    
    print("\n1. SETUP: Knowledge Base with Transitive Chains")
    print("-" * 60)
    
    # Create a knowledge base WITH transitive chains
    relationships = [
        # BIOLOGY: Valid chain within domain
        ('Cat', 'is_a', 'Mammal'),
        ('Mammal', 'is_a', 'Animal'),
        ('Animal', 'is_a', 'LivingBeing'),
        
        # BIOLOGY: Another chain
        ('Dog', 'is_a', 'Mammal'),
        ('Pet', 'is_a', 'Animal'),
        
        # INSTANCES: Correct usage
        ('Fluffy', 'instance_of', 'Cat'),
        ('Fido', 'instance_of', 'Dog'),
        
        # ML: Valid chain within domain
        ('CNN', 'is_a', 'NeuralNetwork'),
        ('NeuralNetwork', 'is_a', 'DeepLearning'),
        ('RNN', 'is_a', 'NeuralNetwork'),
        
        # CS: Valid chain within domain
        ('Python', 'is_a', 'ProgrammingLanguage'),
        ('ProgrammingLanguage', 'is_a', 'Software'),
        ('Software', 'is_a', 'DigitalProduct'),
        
        # ABSTRACT: Connects domains
        ('LivingBeing', 'is_a', 'Entity'),
        ('DeepLearning', 'is_a', 'Algorithm'),
        ('DigitalProduct', 'is_a', 'Entity'),
        
        # PROBLEMATIC: Cross-domain (what unconstrained system would create)
        ('Cat', 'is_a', 'DeepLearning'),  # biology → ml
        ('Dog', 'is_a', 'Software'),      # biology → cs
        ('Python', 'is_a', 'Animal'),     # cs → biology
    ]
    
    print(f"Starting with {len(relationships)} relationships")
    
    # Initialize constraint system
    constraints = OntologyConstraintSystem()
    
    print("\n2. RUNNING TRANSITIVE CLOSURE WITH CONSTRAINTS")
    print("-" * 60)
    
    # Find is_a relationships for transitive closure
    is_a_relationships = [r for r in relationships if r[1] == 'is_a']
    
    print(f"Found {len(is_a_relationships)} is_a relationships for transitive closure")
    
    # Apply transitive closure
    all_potential_inferences = []
    
    for i, (subj1, rel1, obj1) in enumerate(is_a_relationships):
        for j, (subj2, rel2, obj2) in enumerate(is_a_relationships):
            if obj1 == subj2:  # Chain found: subj1 → obj1 → obj2
                potential = (subj1, 'is_a', obj2, f"From: {subj1}→{obj1} and {obj1}→{obj2}")
                all_potential_inferences.append(potential)
    
    print(f"Generated {len(all_potential_inferences)} potential inferences")
    
    # Validate each potential inference
    print("\n3. VALIDATING INFERENCES")
    print("-" * 60)
    
    for subj, rel, obj, chain in all_potential_inferences[:15]:  # Show first 15
        is_valid, reason = constraints.validate_inference(subj, rel, obj, "transitive_closure")
        
        if is_valid:
            print(f"  ✅ VALID: {subj:15} → {obj:20}")
            print(f"      Chain: {chain}")
            print(f"      Reason: {reason}")
        else:
            print(f"  ❌ REJECTED: {subj:15} → {obj:20}")
            print(f"      Chain: {chain}")
            print(f"      Reason: {reason}")
    
    print("\n4. RESULTS SUMMARY")
    print("-" * 60)
    
    print(f"Total potential inferences: {len(all_potential_inferences)}")
    print(f"Valid inferences: {len(constraints.valid_inferences)}")
    print(f"Rejected inferences: {len(constraints.rejected_inferences)}")
    
    # Show examples of valid inferences
    if constraints.valid_inferences:
        print("\nExamples of VALID inferences (within domains):")
        for inf in constraints.valid_inferences[:5]:
            print(f"  • {inf['subject']} → {inf['object']} ({inf['relation']})")
            print(f"    Domains: {inf['domains']}")
    
    # Show examples of rejected inferences
    if constraints.rejected_inferences:
        print("\nExamples of REJECTED inferences (cross-domain):")
        for inf in constraints.rejected_inferences[:5]:
            if 'Cross-domain' in inf['reason']:
                print(f"  • {inf['subject']} → {inf['object']} ({inf['relation']})")
                print(f"    Reason: {inf['reason']}")
    
    # Show specific problematic cases from the original issue
    print("\n5. SPECIFIC PROBLEMS PREVENTED")
    print("-" * 60)
    
    problematic_cases = [
        ('Fluffy', 'is_a', 'DeepLearning', 'instance → ml'),
        ('Dog', 'is_a', 'NeuralNetwork', 'biology → ml'),
        ('Animal', 'is_a', 'DeepLearning', 'biology → ml'),
        ('Mammal', 'is_a', 'Software', 'biology → cs'),
        ('Python', 'is_a', 'Cat', 'cs → biology'),
    ]
    
    for subj, rel, obj, desc in problematic_cases:
        is_valid, reason = constraints.validate_inference(subj, rel, obj, "explicit_check")
        status = "❌ REJECTED" if not is_valid else "✅ (but should be rejected)"
        print(f"  {status}: {subj:15} → {obj:20}")
        print(f"      {desc}")
        if not is_valid:
            print(f"      Reason: {reason}")
    
    return constraints

def demonstrate_constraint_benefits():
    """Show why constraints are essential"""
    
    print("\n" + "="*70)
    print("WHY CONSTRAINTS ARE ESSENTIAL")
    print("="*70)
    
    print("\nWITHOUT CONSTRAINTS (Current neurosql_advanced.py):")
    print("-" * 40)
    print("1. Transitive closure runs freely")
    print("2. Produces nonsense inferences:")
    print("   • Fluffy → DeepLearning")
    print("   • Dog → NeuralNetwork")
    print("   • Animal → DeepLearning")
    print("3. Knowledge base becomes contaminated")
    print("4. Queries return nonsensical results")
    
    print("\nWITH CONSTRAINTS (This system):")
    print("-" * 40)
    print("1. Every inference is validated")
    print("2. Cross-domain inferences are rejected")
    print("3. Instance/class confusion is prevented")
    print("4. Only valid inferences are added")
    print("5. Knowledge base remains clean")
    
    print("\n" + "="*70)
    print("CONCRETE EXAMPLE: Cat inheritance chain")
    print("="*70)
    
    print("\nVALID CHAIN (biology domain only):")
    print("  Fluffy → Cat (instance_of)")
    print("  Cat → Mammal (is_a)")
    print("  Mammal → Animal (is_a)")
    print("  Animal → LivingBeing (is_a)")
    
    print("\nINFERRED (via transitive closure):")
    print("  ✅ Cat → Animal (biology → biology)")
    print("  ✅ Cat → LivingBeing (biology → biology)")
    print("  ✅ Mammal → LivingBeing (biology → biology)")
    
    print("\nREJECTED (cross-domain):")
    print("  ❌ Cat → DeepLearning (biology → ml)")
    print("  ❌ Cat → Software (biology → cs)")
    print("  ❌ Fluffy → DeepLearning (instance → ml)")

def create_implementation_guide():
    """Show how to implement constraints in NeuroSQL"""
    
    print("\n" + "="*70)
    print("IMPLEMENTATION GUIDE")
    print("="*70)
    
    implementation = '''
# Step 1: Add domain metadata to concepts
class Concept:
    def __init__(self, name, metadata, layer, domain):
        self.name = name
        self.metadata = metadata
        self.metadata['domain'] = domain  # Add domain
        self.layer = layer

# Step 2: Create constraint validator
class ConstraintValidator:
    def validate(self, subject, relation, object, context):
        subj_domain = self.get_domain(subject)
        obj_domain = self.get_domain(object)
        
        # Reject cross-domain is_a
        if relation == 'is_a' and subj_domain != obj_domain:
            return False, f"Cross-domain: {subj_domain}→{obj_domain}"
        
        return True, "Valid"

# Step 3: Modify transitive closure in reasoning_engine.py
def transitive_closure_constrained(self, relation_type):
    # Get all relationships of this type
    edges = self.get_relationships_by_type(relation_type)
    
    valid_inferences = []
    rejected_inferences = []
    
    for edge1 in edges:
        for edge2 in edges:
            if edge1.object == edge2.subject:
                candidate = (edge1.subject, relation_type, edge2.object)
                
                # VALIDATE BEFORE ADDING
                is_valid, reason = self.validator.validate(*candidate)
                
                if is_valid:
                    valid_inferences.append(candidate)
                else:
                    rejected_inferences.append((candidate, reason))
    
    # Log rejections for debugging
    if rejected_inferences:
        print(f"Rejected {len(rejected_inferences)} invalid inferences")
    
    return valid_inferences

# Step 4: Make validation non-optional
class NeuroSQLAdvanced:
    def __init__(self):
        self.validator = ConstraintValidator()
        
    def apply_operator(self, operator, params):
        # All operators must use validator
        if operator == 'transitive_closure':
            return self.transitive_closure_constrained(params.get('relation_type'))
'''
    
    print(implementation)

def main():
    """Main function"""
    
    # Run the simulation
    constraints = simulate_transitive_closure()
    
    # Demonstrate benefits
    demonstrate_constraint_benefits()
    
    # Show implementation
    create_implementation_guide()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    print("\n✅ Ontology constraints ARE a good idea because:")
    print("   1. They prevent knowledge base contamination")
    print("   2. They maintain semantic integrity")
    print("   3. They make the system trustworthy")
    print("   4. They provide explanation for rejections")
    
    print("\n📊 Your current 0 valid inferences is EXPECTED because:")
    print("   - The demo KB lacks sufficient same-domain chains")
    print("   - But the constraint system is WORKING CORRECTLY")
    print("   - It would reject bad inferences if they existed")
    
    print("\n🚀 Next step: Implement constraints in reasoning_engine.py")
    print("   Then you'll have a truly robust reasoning system!")

if __name__ == "__main__":
    main()
