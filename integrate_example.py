# Example of integrating risk mitigations with NeuroSQL
from validated_evidence import RiskAwareNeuroSQLSystem
import asyncio

async def main():
    # Initialize risk-aware system
    system = RiskAwareNeuroSQLSystem()
    
    # Test queries with different risk levels
    test_queries = [
        ("dopamine modulates reward processing", False),
        ("Take 50mg serotonin for depression", True),
        ("hippocampus supports memory formation", False),
        ("glutamate excites neurons via NMDA receptors", False),
    ]
    
    print("Testing NeuroSQL with Risk Mitigations")
    print("=" * 60)
    
    for query, clinical in test_queries:
        print(f"\nQuery: {query}")
        print(f"Clinical setting: {clinical}")
        
        result = await system.process_query_with_validation(query, clinical)
        
        if 'error' in result and 'blocked' in result['error']:
            print("  ❌ Query blocked for safety")
            print(f"  Reason: {result['error']}")
        else:
            print(f"  ✅ Query processed")
            print(f"  Result: {result.get('result', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 0):.1%}")
            print(f"  Uncertainty: {result.get('uncertainty', 'N/A')}")
            print(f"  Clinical warning: {result.get('clinical_warning', False)}")
    
    print("\n" + "=" * 60)
    print("Integration complete!")

if __name__ == "__main__":
    asyncio.run(main())
