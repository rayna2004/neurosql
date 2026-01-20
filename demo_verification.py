import asyncio
from neurosql_nextgen_final import UnifiedInferenceEngine
from neuroecho_wrapper import VerifiedInferenceEngine

async def compare_systems():
    print("="*80)
    print("NEUROSQL vs NEUROECHO-VERIFIED COMPARISON")
    print("="*80)
    
    original_engine = UnifiedInferenceEngine()
    verified_engine = VerifiedInferenceEngine()
    
    query = {
        'type': 'transitive',
        'entity': 'dopamine',
        'relation': 'MODULATES',
        'context': {'domain': 'neuropsychiatry'}
    }
    
    print("\n📊 TEST QUERY:")
    print(f"   {query}")
    
    print("\n1️⃣ ORIGINAL NEUROSQL:")
    original_result = await original_engine.infer(query)
    print(f"   Result: {original_result.result}")
    print(f"   Confidence: {original_result.confidence:.1%}")
    print(f"   Evidence: {original_result.evidence}")
    print(f"   ⚠️  No verification - confidence may be random!")
    
    print("\n2️⃣ NEUROECHO-VERIFIED:")
    verified_result = await verified_engine.infer(query)
    print(f"   Result: {verified_result.result}")
    print(f"   Original Confidence: {original_result.confidence:.1%}")
    print(f"   Fidelity Score: {verified_result.fidelity_score:.1%}")
    print(f"   Verification: {'✅ PASSED' if verified_result.verification_passed else '❌ FAILED'}")
    print(f"   Final Confidence: {verified_result.confidence:.1%}")
    
    if verified_result.computation_trace:
        print(f"\n   🔍 COMPUTATION TRACE:")
        print(f"      Nodes: {verified_result.computation_trace['nodes_visited']}")
        print(f"      Edges: {verified_result.computation_trace['edges_traversed']}")
    
    if verified_result.reasoning_pathway:
        print(f"\n   🛤️  REASONING PATHWAY:")
        for step in verified_result.reasoning_pathway:
            print(f"      {step}")
    
    print("\n\n3️⃣ BATCH VERIFICATION TEST:")
    
    test_queries = [
        {'type': 'transitive', 'entity': 'serotonin', 'relation': 'affects'},
        {'type': 'probabilistic', 'evidence': {'data': 'test'}},
        {'type': 'similarity', 'entity': 'glutamate'},
    ]
    
    for i, q in enumerate(test_queries, 1):
        result = await verified_engine.infer(q)
        status = "✅" if result.verification_passed else "❌"
        print(f"   Query {i}: {q['type']:15s} | Fidelity: {result.fidelity_score:.1%} | {status}")
    
    print("\n4️⃣ VERIFICATION STATISTICS:")
    stats = verified_engine.get_verification_stats()
    print(f"   Total verifications: {stats['total_verifications']}")
    print(f"   Passed: {stats['passed']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   Pass rate: {stats['pass_rate']:.1%}")
    print(f"   Average fidelity: {stats['avg_fidelity']:.1%}")
    
    print("\n" + "="*80)
    print("KEY INSIGHT:")
    print("NeuroEcho verifies that confidence scores are grounded in actual reasoning,")
    print("not just random numbers. Failed verification → confidence set to 0%")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(compare_systems())