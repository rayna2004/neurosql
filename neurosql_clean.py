#!/usr/bin/env python3
"""NeuroSQL Clean Demo - With proper ontology enforcement"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the ontology guard
try:
    from ontology_guard import OntologyGuard
    
    print("="*70)
    print("NEUROSQL CLEAN DEMO - ONTOLOGY PROTECTION")
    print("="*70)
    print()
    
    # Create ontology guard instance
    guard = OntologyGuard()
    
    # Test some valid relationships
    print("Testing VALID relationships:")
    print("-" * 40)
    
    valid_tests = [
        ("dopamine", "neuron", "neurotransmitter"),
        ("hippocampus", "memory", "function"),
        ("action_potential", "neuron", "generated_by"),
        ("glutamate", "excitatory", "is_type")
    ]
    
    for subject, object, relation in valid_tests:
        is_valid, reason = guard.validate_relationship(subject, object, relation)
        print(f"✓ {subject} → {object} ({relation}): {reason}")
    
    print()
    print("Testing INVALID relationships (should be rejected):")
    print("-" * 40)
    
    invalid_tests = [
        ("dopamine", "happiness", "causes"),
        ("neuron", "memory", "stores"),
        ("synapse", "thought", "contains")
    ]
    
    for subject, object, relation in invalid_tests:
        is_valid, reason = guard.validate_relationship(subject, object, relation)
        print(f"✗ {subject} → {object} ({relation}): {reason}")
    
    print()
    print("="*70)
    print("DEMO COMPLETE")
    print("="*70)
    
except ImportError as e:
    print(f"Error: Could not import ontology_guard. {e}")
    print("Make sure ontology_guard.py exists in the current directory.")
except Exception as e:
    print(f"Error: {e}")
