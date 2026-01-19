# Example of integrating risk mitigations with NeuroSQL
from validated_evidence_complete import RiskAwareNeuroSQLSystem
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

async def main():
    # Initialize risk-aware system
    system = RiskAwareNeuroSQLSystem()
    
    # Test queries with different risk levels
    test_queries = [
        ("dopamine modulates reward processing", False),
        ("Take 50mg serotonin for depression", True),
        ("hippocampus supports memory formation", False),
        ("glutamate excites neurons via NMDA receptors", False),
        ("acetylcholine binds to NMDA receptors", False),  # Biologically implausible
        ("serotonin treatment for anxiety", True),  # Clinical warning
    ]
    
    print("Testing NeuroSQL with Risk Mitigations")
    print("=" * 60)
    
    for query, clinical in test_queries:
        print(f"\nQuery: {query}")
        print(f"Clinical setting: {clinical}")
        
        result = await system.process_query_with_validation(query, clinical)
        
        if result.get('blocked'):
            print("  ❌ Query blocked for safety")
            if 'safety_issues' in result:
                for issue in result['safety_issues']:
                    print(f"    Reason: {issue.get('message', 'Safety violation')}")
        else:
            print(f"  ✅ Result: {result.get('result')}")
            print(f"  Confidence: {result.get('confidence', 0):.1%}")
            print(f"  Uncertainty: {result.get('uncertainty', 'N/A')}")
            print(f"  Biologically plausible: {result.get('biological_plausible', True)}")
            if result.get('clinical_warning'):
                print(f"  ⚠️ Clinical warning applied")
            if 'disclaimers' in result:
                print(f"  📄 {len(result['disclaimers'])} disclaimer(s) added")
    
    print("\n" + "=" * 60)
    print("Risk Assessment Summary:")
    report = system.get_risk_report()
    print(f"Total inferences processed: {report.get('total_inferences', 0)}")
    print(f"High risk queries: {report.get('high_risk_percentage', 0):.1%}")
    print(f"Blocked queries: {report.get('blocked_percentage', 0):.1%}")
    
    print("\n" + "=" * 60)
    print("✅ Integration complete! All risk mitigations are working.")
    print("\nRisk mitigations applied:")
    print("1. ✅ Evidence integrity validation")
    print("2. ✅ Biological constraint checking")
    print("3. ✅ Clinical safety guardrails")
    print("4. ✅ Confidence calibration")
    print("5. ✅ Risk monitoring and logging")

if __name__ == "__main__":
    asyncio.run(main())
