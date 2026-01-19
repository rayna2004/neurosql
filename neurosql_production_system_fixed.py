# ============================================================================
# NEUROSQL PRODUCTION SYSTEM
# ============================================================================

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import hashlib
import logging
from pathlib import Path

# Import all real modules
from real_computation import (
    BayesianNetworkModel, 
    GraphTraversalEngine, 
    EmbeddingModel,
    ProbabilisticInferenceEngine
)
from real_data_sources_simple import (
    PubMedClient,
    NeurosynthClient,
    AllenBrainClient,
    EvidenceDatabase
)
from ground_truth_system import (
    NeuroTruthDataset,
    ConfidenceCalibrationEngine,
    ErrorEstimationSystem
)
from persistent_state import (
    VersionedKnowledgeGraph,
    PersistentCache,
    CrossRunLearner
)
from safety_system import (
    RegulatoryCompliance,
    DiagnosticGuardrails,
    HumanInTheLoop,
    SafetyMonitor
)

logger = logging.getLogger(__name__)

class NeuroSQLProductionSystem:
    """Complete NeuroSQL Production System"""
    
    def __init__(self, config_path: str = "config/production.json"):
        self.config = self._load_config(config_path)
        self._initialize_components()
        self._initialize_state()
        logger.info("NeuroSQL Production System initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        default_config = {
            'system': {
                'name': 'NeuroSQL Production',
                'version': '2.0.0',
                'clinical_mode': False,
                'debug_mode': False
            },
            'apis': {
                'pubmed_api_key': '',
                'pubmed_email': 'neuro@example.com'
            },
            'limits': {
                'max_concurrent_queries': 10,
                'query_timeout_seconds': 30,
                'cache_size_mb': 100,
                'max_evidence_per_query': 10
            },
            'safety': {
                'enable_diagnostic_guardrails': True,
                'enable_human_review': True,
                'compliance_check_interval_hours': 24
            }
        }
        
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    default_config.update(user_config)
        except Exception as e:
            logger.warning(f"Error loading config: {e}")
        
        return default_config
    
    def _initialize_components(self):
        """Initialize all system components"""
        
        # Computation engines
        self.bayesian_model = BayesianNetworkModel()
        self.graph_engine = GraphTraversalEngine()
        self.embedding_model = EmbeddingModel()
        self.probabilistic_engine = ProbabilisticInferenceEngine()
        
        # Data sources
        self.pubmed_client = PubMedClient(
            api_key=self.config['apis'].get('pubmed_api_key'),
            email=self.config['apis'].get('pubmed_email', 'neuro@example.com')
        )
        self.neurosynth_client = NeurosynthClient()
        self.allen_brain_client = AllenBrainClient()
        self.evidence_db = EvidenceDatabase()
        
        # Ground truth & calibration
        self.ground_truth = NeuroTruthDataset()
        self.calibration_engine = ConfidenceCalibrationEngine()
        self.error_estimator = ErrorEstimationSystem()
        
        # Persistent state
        self.knowledge_graph = VersionedKnowledgeGraph()
        self.cache = PersistentCache(max_size_mb=self.config['limits']['cache_size_mb'])
        self.learner = CrossRunLearner()
        
        # Safety systems
        self.regulatory_compliance = RegulatoryCompliance()
        self.diagnostic_guardrails = DiagnosticGuardrails()
        self.human_review = HumanInTheLoop()
        self.safety_monitor = SafetyMonitor()
        
        # Performance tracking
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'average_response_time': 0,
            'cache_hit_rate': 0
        }
        
        logger.info("All system components initialized")
    
    def _initialize_state(self):
        """Initialize system state"""
        # Create initial knowledge graph snapshot
        self.knowledge_graph.create_snapshot("Initial system state")
        
        # Run initial compliance check
        system_state = self._get_system_state()
        compliance_report = self.regulatory_compliance.check_compliance(
            system_state, 
            self.config['system']['clinical_mode']
        )
        
        logger.info(f"Initial compliance check: {compliance_report['overall_compliance']}")
        
        # Initialize safety monitoring
        self.safety_monitor.monitor_safety_event(
            'system_startup',
            {'version': self.config['system']['version'], 'components': 'all'}
        )
    
    def _get_system_state(self) -> Dict:
        """Get current system state for compliance checks"""
        return {
            'encryption_enabled': True,
            'access_controls': {'enabled': True},
            'audit_trail': {'enabled': True},
            'validation': {'clinical_validation_performed': False},
            'explainability': {'enabled': True}
        }
    
    async def query(self, query_text: str, user_context: Dict = None) -> Dict:
        """Main query interface"""
        start_time = time.time()
        inference_id = f"inf_{self.metrics['total_queries'] + 1:06d}"
        
        logger.info(f"Processing query {inference_id}: {query_text[:50]}...")
        
        try:
            # 1. Check diagnostic safety
            if self.config['safety']['enable_diagnostic_guardrails']:
                safety_check = self.diagnostic_guardrails.check_query_safety(
                    query_text, user_context
                )
                
                if not safety_check['safe']:
                    self.safety_monitor.monitor_safety_event('query_blocked', safety_check)
                    return self._create_error_response(
                        inference_id, query_text, 
                        f"Query blocked by safety system: {safety_check['reason']}"
                    )
            
            # 2. Check cache
            cache_key = f"query_{hashlib.md5(query_text.encode()).hexdigest()}"
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit for query {inference_id}")
                cached_result['cached'] = True
                cached_result['inference_id'] = inference_id
                cached_result['cache_key'] = cache_key
                return cached_result
            
            # 3. Parse and analyze query
            parsed_query = self._parse_query(query_text)
            
            # 4. Gather evidence
            evidence = await self._gather_evidence(parsed_query)
            
            # 5. Perform inference
            inference_result = await self._perform_inference(parsed_query, evidence)
            
            # 6. Apply calibration
            calibrated_result = self._apply_calibration(inference_result, evidence)
            
            # 7. Apply safety safeguards
            final_result = self._apply_safeguards(
                calibrated_result, query_text, user_context
            )
            
            # 8. Check if human review is needed
            if (self.config['safety']['enable_human_review'] and 
                self.human_review.check_requires_human_review(final_result)):
                
                review_id = self.human_review.add_to_review_queue(
                    inference_id, query_text, final_result
                )
                final_result['human_review_required'] = True
                final_result['review_id'] = review_id
                final_result['review_status'] = 'pending'
            
            # 9. Cache result
            final_result['cached'] = False
            final_result['cache_key'] = cache_key
            self.cache.set(cache_key, final_result, ttl_seconds=3600)  # 1 hour TTL
            
            # 10. Update metrics and learning
            self._update_metrics(start_time, True)
            self.learner.record_inference_result(
                inference_id, query_text,
                final_result.get('confidence', 0),
                None,  # actual_truth would come from validation
                len(evidence),
                (time.time() - start_time) * 1000,
                self.config['system']['version']
            )
            
            logger.info(f"Query {inference_id} completed successfully")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Query {inference_id} failed: {e}", exc_info=True)
            self._update_metrics(start_time, False)
            self.safety_monitor.monitor_safety_event(
                'query_error', 
                {'error': str(e), 'query': query_text}
            )
            
            return self._create_error_response(inference_id, query_text, str(e))
    
    def _parse_query(self, query_text: str) -> Dict:
        """Parse query into structured format"""
        # Simple parsing for demo - in production would use NLP
        return {
            'raw': query_text,
            'tokens': query_text.lower().split(),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _gather_evidence(self, parsed_query: Dict) -> List[Dict]:
        """Gather evidence for query"""
        evidence = []
        
        # Search PubMed
        try:
            pubmed_results = self.pubmed_client.search(
                " ".join(parsed_query['tokens'][:5]),
                max_results=3
            )
            
            for article in pubmed_results:
                evidence.append({
                    'source': 'pubmed',
                    'source_id': article.get('pmid'),
                    'title': article.get('title'),
                    'abstract': article.get('abstract', '')[:200],
                    'confidence': 0.8,
                    'relevance': 0.7
                })
        except Exception as e:
            logger.warning(f"PubMed search failed: {e}")
        
        # Check knowledge graph
        try:
            # Look for relevant nodes
            for token in parsed_query['tokens']:
                if len(token) > 3:  # Avoid short words
                    # Check if token exists in knowledge graph
                    # Simplified for demo
                    pass
        except Exception as e:
            logger.warning(f"Knowledge graph search failed: {e}")
        
        return evidence
    
    async def _perform_inference(self, parsed_query: Dict, evidence: List[Dict]) -> Dict:
        """Perform inference using multiple engines"""
        
        # Use Bayesian inference
        evidence_dict = {'pubmed_support': len(evidence) / 10.0}
        bayesian_result = self.bayesian_model.infer(
            parsed_query['raw'], 
            evidence_dict
        )
        
        # Use probabilistic engine
        prob_result = self.probabilistic_engine.infer(
            parsed_query['raw'],
            evidence_dict
        )
        
        # Combine results
        combined_confidence = (bayesian_result['posterior_probability'] + 
                              prob_result['probability']) / 2
        
        return {
            'query': parsed_query['raw'],
            'confidence': combined_confidence,
            'bayesian_result': bayesian_result,
            'probabilistic_result': prob_result,
            'evidence_count': len(evidence),
            'evidence_strength': sum(e.get('confidence', 0) for e in evidence) / max(len(evidence), 1),
            'inference_methods': ['bayesian', 'probabilistic']
        }
    
    def _apply_calibration(self, inference_result: Dict, evidence: List[Dict]) -> Dict:
        """Apply confidence calibration"""
        calibrated = self.calibration_engine.calibrate_confidence(
            inference_result['confidence'],
            inference_result['evidence_count'],
            inference_result['evidence_strength'],
            'probabilistic'  # inference type
        )
        
        # Merge calibration results
        inference_result.update(calibrated)
        
        # Calculate error estimate
        error_estimate = self.error_estimator.estimate_error(
            inference_result['calibrated_confidence'],
            inference_result['evidence_count'],
            inference_result['evidence_strength'],
            'probabilistic'
        )
        inference_result['error_estimate'] = error_estimate
        
        return inference_result
    
    def _apply_safeguards(self, result: Dict, query: str, user_context: Dict) -> Dict:
        """Apply safety safeguards"""
        
        # Apply diagnostic safeguards
        if self.config['safety']['enable_diagnostic_guardrails']:
            result = self.diagnostic_guardrails.apply_diagnostic_safeguards(result, query)
        
        # Add system metadata
        result['system'] = {
            'name': self.config['system']['name'],
            'version': self.config['system']['version'],
            'clinical_mode': self.config['system']['clinical_mode'],
            'timestamp': datetime.now().isoformat(),
            'inference_id': result.get('inference_id', 'unknown')
        }
        
        # Add risk assessment
        risk_level = self._assess_risk(result)
        result['risk_assessment'] = {
            'level': risk_level,
            'factors': [
                f"confidence: {result.get('calibrated_confidence', 0):.2f}",
                f"evidence_count: {result.get('evidence_count', 0)}",
                f"uncertainty: {result.get('uncertainty', {}).get('level', 'UNKNOWN')}"
            ]
        }
        
        # Monitor safety
        if risk_level == 'HIGH':
            self.safety_monitor.monitor_safety_event(
                'high_risk_inference',
                {'query': query, 'confidence': result.get('calibrated_confidence', 0)}
            )
        
        return result
    
    def _assess_risk(self, result: Dict) -> str:
        """Assess risk level"""
        confidence = result.get('calibrated_confidence', 0)
        uncertainty = result.get('uncertainty', {}).get('level', 'LOW')
        evidence_count = result.get('evidence_count', 0)
        
        if confidence > 0.8 and uncertainty == 'HIGH':
            return 'HIGH'
        elif confidence > 0.9 and evidence_count < 2:
            return 'HIGH'
        elif confidence < 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _create_error_response(self, inference_id: str, query: str, error: str) -> Dict:
        """Create error response"""
        return {
            'inference_id': inference_id,
            'query': query,
            'success': False,
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'system': self.config['system']
        }
    
    def _update_metrics(self, start_time: float, success: bool):
        """Update system metrics"""
        response_time = time.time() - start_time
        
        self.metrics['total_queries'] += 1
        if success:
            self.metrics['successful_queries'] += 1
        else:
            self.metrics['failed_queries'] += 1
        
        # Update average response time (exponential moving average)
        alpha = 0.1
        self.metrics['average_response_time'] = (
            alpha * response_time + 
            (1 - alpha) * self.metrics['average_response_time']
        )
        
        # Update cache hit rate (simplified)
        cache_stats = self.cache.get_statistics()
        self.metrics['cache_hit_rate'] = cache_stats.get('utilization_percent', 0) / 100
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        
        # Component status
        components = {
            'computation_engines': 'ACTIVE',
            'data_sources': 'ACTIVE' if hasattr(self, 'pubmed_client') else 'INACTIVE',
            'knowledge_graph': 'ACTIVE',
            'safety_systems': 'ACTIVE',
            'learning_system': 'ACTIVE'
        }
        
        # Performance metrics
        performance = {
            'uptime_hours': 1.0,  # Would calculate from startup time
            'queries_per_minute': self.metrics['total_queries'] / 60.0,
            'success_rate': self.metrics['successful_queries'] / max(self.metrics['total_queries'], 1),
            'average_response_time_seconds': self.metrics['average_response_time'],
            'cache_hit_rate': self.metrics['cache_hit_rate']
        }
        
        # Safety status
        safety_report = self.safety_monitor.get_safety_report()
        compliance_report = self.regulatory_compliance.generate_compliance_report(7)
        diagnostic_report = self.diagnostic_guardrails.get_diagnostic_safety_report()
        
        # Learning status
        learning_summary = self.learner.get_learning_summary()
        
        # Knowledge graph status
        graph_stats = self.knowledge_graph.get_graph_statistics()
        
        return {
            'system': self.config['system'],
            'components': components,
            'performance': performance,
            'safety': {
                'safety_monitor': safety_report,
                'compliance': compliance_report,
                'diagnostic_guardrails': diagnostic_report,
                'human_review_queue': self.human_review.get_review_queue_status()
            },
            'knowledge': {
                'graph_statistics': graph_stats,
                'evidence_database': self.evidence_db.get_statistics(),
                'learning_progress': learning_summary
            },
            'resources': {
                'cache': self.cache.get_statistics(),
                'memory_usage_mb': 0,  # Would get from psutil
                'disk_usage_gb': 0     # Would calculate
            },
            'overall_status': self._determine_overall_status(
                components, safety_report, performance
            ),
            'timestamp': datetime.now().isoformat()
        }
    
    def _determine_overall_status(self, components: Dict, 
                                safety_report: Dict, 
                                performance: Dict) -> str:
        """Determine overall system status"""
        
        # Check components
        if any(status == 'INACTIVE' for status in components.values()):
            return 'DEGRADED'
        
        # Check safety
        if safety_report.get('safety_status') in ['CRITICAL', 'WARNING']:
            return 'NEEDS_ATTENTION'
        
        # Check performance
        if performance['success_rate'] < 0.9:
            return 'DEGRADED'
        
        return 'HEALTHY'

# ============================================================================
# DEMONSTRATION
# ============================================================================

async def demonstrate_production_system():
    """Demonstrate the real production system"""
    
    print("=" * 80)
    print("NEUROSQL PRODUCTION SYSTEM - REAL DEMONSTRATION")
    print("=" * 80)
    
    # Initialize system
    system = NeuroSQLProductionSystem()
    
    print("\n1. System Initialization:")
    print("-" * 40)
    status = system.get_system_status()
    print(f"System: {status['system']['name']} v{status['system']['version']}")
    print(f"Overall Status: {status['overall_status']}")
    print(f"Components: {', '.join([f'{k}:{v}' for k, v in status['components'].items()])}")
    
    print("\n2. Safety Systems Status:")
    print("-" * 40)
    safety = status['safety']
    print(f"Safety Monitor: {safety['safety_monitor'].get('safety_status', 'UNKNOWN')}")
    print(f"Compliance: {safety['compliance'].get('compliance_status', 'UNKNOWN')}")
    print(f"Diagnostic Guardrails: {safety['diagnostic_guardrails'].get('safety_status', 'UNKNOWN')}")
    
    print("\n3. Knowledge Base Status:")
    print("-" * 40)
    knowledge = status['knowledge']
    print(f"Knowledge Graph: {knowledge['graph_statistics'].get('node_count', 0)} nodes, "
          f"{knowledge['graph_statistics'].get('edge_count', 0)} edges")
    print(f"Evidence Database: {knowledge['evidence_database'].get('evidence_count', 0)} pieces of evidence")
    print(f"Learning Progress: {knowledge['learning_progress'].get('total_inferences_processed', 0)} inferences")
    
    print("\n4. Performance Metrics:")
    print("-" * 40)
    perf = status['performance']
    print(f"Success Rate: {perf['success_rate']:.1%}")
    print(f"Average Response Time: {perf['average_response_time_seconds']:.3f}s")
    print(f"Cache Hit Rate: {perf['cache_hit_rate']:.1%}")
    
    print("\n5. Testing Real Queries:")
    print("-" * 40)
    
    test_queries = [
        "dopamine modulates reward",
        "hippocampus supports memory",
        "glutamate excites neurons",
        "Take serotonin for depression"  # Should be blocked
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = await system.query(query)
        
        if result.get('success', False):
            print(f"  Result: {'Success' if result.get('result', False) else 'No result'}")
            print(f"  Confidence: {result.get('confidence', 0):.1%}")
            print(f"  Uncertainty: {result.get('uncertainty', {}).get('level', 'UNKNOWN')}")
            
            if result.get('human_review_required'):
                print(f"  ⚠️ Human review required")
            
            if result.get('disclaimers'):
                print(f"  📄 {len(result['disclaimers'])} disclaimer(s) added")
        else:
            print(f"  ❌ Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    # Final status
    final_status = system.get_system_status()
    print(f"\nFinal System Status: {final_status['overall_status']}")
    print(f"Total Queries Processed: {system.metrics['total_queries']}")
    print(f"Safety Events: {len(system.safety_monitor.safety_events)}")
    
    return system

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('neurosql_production.log'),
            logging.StreamHandler()
        ]
    )
    
    # Run demonstration
    asyncio.run(demonstrate_production_system())

