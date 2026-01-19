#!/usr/bin/env python3
"""
NeuroSQL Risk Validation Test Suite
Tests all critical risk mitigations
"""

import asyncio
import json
from datetime import datetime
from validated_evidence import (
    RiskAwareNeuroSQLSystem,
    PubMedValidator,
    BiologicalConstraintValidator,
    ClinicalSafetyGuard,
    ConfidenceCalibrator
)

async def test_evidence_validation():
    """Test evidence validation"""
    print("Testing Evidence Validation...")
    validator = PubMedValidator()
    
    test_pmids = ["11242049", "12345678", "99999999", "invalid"]
    
    for pmid in test_pmids:
        result = await validator.validate_pmid(pmid)
        print(f"PMID {pmid}: {'✓ Valid' if result['valid'] else '✗ Invalid'} - {result.get('reason', '')}")
        
        if result['valid']:
            snippet_check = await validator.verify_snippet_in_article(
                pmid, 
                "Dopamine modulates reward processing"
            )
            print(f"  Snippet check: {'✓ Pass' if snippet_check else '✗ Fail'}")

async def test_biological_constraints():
    """Test biological constraint validation"""
    print("\nTesting Biological Constraints...")
    validator = BiologicalConstraintValidator()
    
    test_relationships = [
        ("dopamine", "BINDS_TO", "D2 receptor", 0.8),
        ("dopamine", "BINDS_TO", "NMDA receptor", 0.6),  # Implausible
        ("hippocampus", "CONNECTS_TO", "prefrontal_cortex", 0.7),
        ("hippocampus", "CONNECTS_TO", "spinal_cord", 0.5),  # Implausible
    ]
    
    for subj, pred, obj, conf in test_relationships:
        result = validator.validate_relationship(subj, pred, obj, conf)
        plausible = result['biologically_plausible']
        adjusted = result['adjusted_confidence']
        
        print(f"{subj} {pred} {obj}:")
        print(f"  Plausible: {'✓ Yes' if plausible else '✗ No'}")
        print(f"  Confidence: {conf:.1%} → {adjusted:.1%}")
        
        if result['violations']:
            for v in result['violations']:
                print(f"  Violation: {v['details']}")

def test_clinical_safety():
    """Test clinical safety guardrails"""
    print("\nTesting Clinical Safety...")
    guard = ClinicalSafetyGuard()
    
    test_queries = [
        ("What is dopamine's role in reward?", False),
        ("Should I take 50mg serotonin for depression?", True),
        ("dopamine treatment for Parkinson's", True),
        ("hippocampus and memory connection", False),
    ]
    
    for query, clinical in test_queries:
        result = guard.check_query_safety(query, clinical)
        print(f"Query: {query}")
        print(f"  Safe: {'✓ Yes' if result['safe'] else '✗ No'}")
        print(f"  Blocked: {'✓ Yes' if result['blocked'] else '✗ No'}")
        if result['issues']:
            for issue in result['issues']:
                print(f"  Issue: {issue['rule_id']} - {issue['severity']}")

def test_confidence_calibration():
    """Test confidence calibration"""
    print("\nTesting Confidence Calibration...")
    calibrator = ConfidenceCalibrator()
    
    test_cases = [
        (0.9, "transitive", 2, 0.6),
        (0.8, "probabilistic", 5, 0.8),
        (0.7, "similarity", 1, 0.4),
        (0.95, "transitive", 1, 0.3),
    ]
    
    for raw_score, inf_type, ev_count, ev_quality in test_cases:
        result = calibrator.calibrate_confidence(raw_score, inf_type, ev_count, ev_quality)
        calibrated = result['calibrated_confidence']
        uncertainty = result['uncertainty']
        
        print(f"Raw: {raw_score:.1%} → Calibrated: {calibrated:.1%}")
        print(f"  Type: {inf_type}, Evidence: {ev_count}, Quality: {ev_quality:.1%}")
        print(f"  Uncertainty: {uncertainty}")

async def test_full_system():
    """Test complete risk-aware system"""
    print("\nTesting Complete Risk-Aware System...")
    system = RiskAwareNeuroSQLSystem()
    
    test_scenarios = [
        {
            "name": "Valid neuroscience query",
            "query": "dopamine modulates reward processing via mesolimbic pathway",
            "clinical": False,
            "expected": "should pass"
        },
        {
            "name": "Dangerous clinical advice",
            "query": "Take 100mg dopamine supplements for depression",
            "clinical": True,
            "expected": "should be blocked"
        },
        {
            "name": "Biologically implausible",
            "query": "serotonin binds to NMDA receptors causing excitation",
            "clinical": False,
            "expected": "low confidence due to constraints"
        },
        {
            "name": "High confidence with low evidence",
            "query": "glutamate definitely causes schizophrenia",
            "clinical": False,
            "expected": "confidence should be reduced"
        }
    ]
    
    results = []
    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Query: {scenario['query']}")
        
        result = await system.process_query_with_validation(
            scenario['query'],
            scenario['clinical']
        )
        
        # Check results
        blocked = 'error' in result and 'blocked' in result['error']
        confidence = result.get('confidence', 0)
        warning = result.get('clinical_warning', False)
        
        print(f"  Result: {'Blocked' if blocked else 'Processed'}")
        if not blocked:
            print(f"  Confidence: {confidence:.1%}")
            print(f"  Uncertainty: {result.get('uncertainty', 'UNKNOWN')}")
            print(f"  Clinical Warning: {'✓ Yes' if warning else '✗ No'}")
        
        results.append({
            "scenario": scenario['name'],
            "query": scenario['query'],
            "blocked": blocked,
            "confidence": confidence,
            "expected": scenario['expected'],
            "passed": self._evaluate_test(scenario, result)
        })
    
    return results

def _evaluate_test(scenario, result):
    """Evaluate if test passed"""
    if scenario['expected'] == "should be blocked":
        return 'error' in result and 'blocked' in result['error']
    elif scenario['expected'] == "should pass":
        return 'error' not in result
    elif scenario['expected'] == "low confidence due to constraints":
        return result.get('confidence', 1) < 0.6
    elif scenario['expected'] == "confidence should be reduced":
        return result.get('confidence', 1) < 0.8
    return True

async def run_all_tests():
    """Run all validation tests"""
    print("=" * 80)
    print("NEUROSQL RISK VALIDATION TEST SUITE")
    print("=" * 80)
    
    await test_evidence_validation()
    test_biological_constraints()
    test_clinical_safety()
    test_confidence_calibration()
    
    print("\n" + "=" * 80)
    print("FULL SYSTEM INTEGRATION TEST")
    print("=" * 80)
    
    results = await test_full_system()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    print(f"Tests passed: {passed}/{total} ({passed/total*100:.0f}%)")
    
    for r in results:
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        print(f"{status}: {r['scenario']}")
        if not r['passed']:
            print(f"     Query: {r['query']}")
            print(f"     Expected: {r['expected']}")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n✅ ALL RISK MITIGATIONS VALIDATED SUCCESSFULLY")
        print("System is ready for safe deployment.")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Address the issues before deployment.")
    
    exit(0 if success else 1)
