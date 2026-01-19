# neurosql_robust.py
"""
NeuroSQL Production System with Comprehensive Error Handling
"""

import sys
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neurosql_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QueryStatus(Enum):
    """Status of a query execution"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    SYSTEM_ERROR = "system_error"

class NeuroSQLError(Exception):
    """Base exception for NeuroSQL errors"""
    pass

class QueryTimeoutError(NeuroSQLError):
    """Raised when a query times out"""
    pass

class InvalidQueryError(NeuroSQLError):
    """Raised when a query is invalid"""
    pass

class QueryResult:
    """Represents the result of a query with full error handling"""
    
    def __init__(self, query_text: str, query_id: str):
        self.query_text = query_text
        self.query_id = query_id
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.status: QueryStatus = QueryStatus.SUCCESS
        self.result_data: Optional[Dict] = None
        self.error_message: Optional[str] = None
        self.error_type: Optional[str] = None
        self.traceback: Optional[str] = None
        self.confidence: float = 0.0
        self.metadata: Dict[str, Any] = {}
    
    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def is_success(self) -> bool:
        """Check if query was successful"""
        return self.status == QueryStatus.SUCCESS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.query_id,
            "query": self.query_text,
            "status": self.status.value,
            "success": self.is_success,
            "result": self.result_data,
            "error": self.error_message,
            "error_type": self.error_type,
            "elapsed": self.elapsed_time,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }

class NeuroSQLRobust:
    """NeuroSQL with comprehensive error handling"""
    
    def __init__(self, timeout_seconds: float = 5.0, max_retries: int = 2):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.query_counter = 0
        self.total_time = 0.0
        self.success_count = 0
        self.failure_count = 0
        self.results: List[QueryResult] = []
        
        # Initialize components with error handling
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize system components with error handling"""
        try:
            # Initialize internal components
            self._initialized = True
            logger.info("NeuroSQL system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NeuroSQL: {e}")
            self._initialized = False
            raise NeuroSQLError(f"System initialization failed: {e}")
    
    def validate_query(self, query_text: str) -> Tuple[bool, Optional[str]]:
        """Validate query before processing"""
        if not query_text or not isinstance(query_text, str):
            return False, "Query must be a non-empty string"
        
        if len(query_text.strip()) == 0:
            return False, "Query cannot be empty or whitespace only"
        
        if len(query_text) > 1000:
            return False, "Query too long (max 1000 characters)"
        
        # Add any specific validation rules here
        banned_patterns = ["DROP", "DELETE", "UPDATE", "INSERT"]
        if any(pattern in query_text.upper() for pattern in banned_patterns):
            return False, "Query contains prohibited patterns"
        
        return True, None
    
    def _simulate_processing_with_errors(self, query_text: str) -> Dict[str, Any]:
        """Simulate processing that can fail in various ways"""
        import random
        
        # Simulate different types of errors
        error_chance = random.random()
        
        if error_chance < 0.1:  # 10% chance of timeout
            time.sleep(6)  # Longer than timeout
            raise QueryTimeoutError(f"Query timed out after {self.timeout_seconds} seconds")
        elif error_chance < 0.2:  # 10% chance of invalid query
            raise InvalidQueryError("Query syntax is invalid")
        elif error_chance < 0.25:  # 5% chance of division by zero
            result = 1 / 0
        elif error_chance < 0.3:  # 5% chance of key error
            data = {}
            return data["nonexistent_key"]
        
        # Normal processing
        time.sleep(0.001 * random.uniform(0.5, 2.0))
        
        # Generate realistic results
        confidence = 0.8 + random.random() * 0.2
        
        return {
            "query": query_text,
            "confidence": confidence,
            "explanation": f"Found {random.randint(1, 5)} relevant papers for '{query_text}'",
            "sources": [
                f"PubMed:{random.randint(10000, 99999)}",
                f"arXiv:{random.randint(2000, 2024)}.{random.randint(1000, 9999)}"
            ],
            "entities": ["neuroscience", "biology", "psychology"][:random.randint(1, 3)],
            "processed_at": datetime.now().isoformat()
        }
    
    def execute_query_with_retry(self, query_text: str, query_id: Optional[str] = None) -> QueryResult:
        """Execute query with retry logic and comprehensive error handling"""
        
        # Generate query ID if not provided
        if query_id is None:
            self.query_counter += 1
            query_id = f"qry_{self.query_counter:06d}"
        
        # Create result object
        result = QueryResult(query_text, query_id)
        result.start_time = time.perf_counter()
        
        try:
            # Validate query
            is_valid, validation_error = self.validate_query(query_text)
            if not is_valid:
                result.status = QueryStatus.INVALID_INPUT
                result.error_message = validation_error
                result.error_type = "ValidationError"
                logger.warning(f"Invalid query {query_id}: {validation_error}")
                return result
            
            logger.info(f"Processing query {query_id}: '{query_text[:50]}...'")
            
            # Try with retries
            for attempt in range(self.max_retries + 1):
                try:
                    # Set timeout
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise QueryTimeoutError(f"Query timeout after {self.timeout_seconds} seconds")
                    
                    # Note: signal handlers don't work well in threads, using simpler approach
                    start_attempt = time.perf_counter()
                    
                    # Execute query
                    query_result = self._simulate_processing_with_errors(query_text)
                    
                    # Check for timeout
                    if time.perf_counter() - start_attempt > self.timeout_seconds:
                        raise QueryTimeoutError(f"Query exceeded timeout of {self.timeout_seconds}s")
                    
                    # Success!
                    result.status = QueryStatus.SUCCESS
                    result.result_data = query_result
                    result.confidence = query_result.get("confidence", 0.0)
                    result.metadata.update({
                        "attempts": attempt + 1,
                        "retry_used": attempt > 0
                    })
                    
                    self.success_count += 1
                    logger.info(f"Query {query_id} succeeded on attempt {attempt + 1}")
                    break
                    
                except (QueryTimeoutError, InvalidQueryError) as e:
                    # Specific errors we don't retry
                    result.status = QueryStatus.TIMEOUT if isinstance(e, QueryTimeoutError) else QueryStatus.INVALID_INPUT
                    result.error_message = str(e)
                    result.error_type = type(e).__name__
                    result.traceback = traceback.format_exc()
                    self.failure_count += 1
                    logger.warning(f"Query {query_id} failed with {type(e).__name__}: {e}")
                    break
                    
                except Exception as e:
                    # Other errors - retry if we have attempts left
                    if attempt < self.max_retries:
                        wait_time = 0.1 * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Query {query_id} failed on attempt {attempt + 1}: {e}. Retrying in {wait_time:.2f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # No more retries
                        result.status = QueryStatus.FAILED
                        result.error_message = str(e)
                        result.error_type = type(e).__name__
                        result.traceback = traceback.format_exc()
                        self.failure_count += 1
                        logger.error(f"Query {query_id} failed after {self.max_retries + 1} attempts: {e}")
                        break
            
        except Exception as e:
            # Catch any unexpected errors in the error handling itself
            result.status = QueryStatus.SYSTEM_ERROR
            result.error_message = f"Unexpected system error: {str(e)}"
            result.error_type = "SystemError"
            result.traceback = traceback.format_exc()
            self.failure_count += 1
            logger.critical(f"Critical error processing query {query_id}: {e}")
        
        finally:
            # Always record end time
            result.end_time = time.perf_counter()
            elapsed = result.elapsed_time
            self.total_time += elapsed
            
            # Log completion
            status_icon = "✅" if result.is_success else "❌"
            logger.info(f"Query {query_id} completed in {elapsed:.6f}s {status_icon}")
        
        return result
    
    def run_comprehensive_test(self):
        """Run a comprehensive test with various query types"""
        
        print()
        print("=" * 70)
        print("NEUROSQL ROBUST SYSTEM - COMPREHENSIVE ERROR HANDLING TEST")
        print("=" * 70)
        print()
        
        # Test queries including edge cases
        test_queries = [
            # Normal queries
            ("dopamine modulates reward", None),
            ("hippocampus supports memory", "custom_id_001"),
            
            # Edge cases that might cause errors
            ("", "empty_query"),  # Empty query
            ("   ", "whitespace_only"),  # Whitespace only
            ("x" * 2000, "too_long_query"),  # Too long
            
            # More normal queries
            ("glutamate excites neurons", None),
            ("gaba inhibits neuronal firing", "neurotransmitter_query"),
            ("serotonin regulates mood", None),
            
            # Queries with potential issues
            ("SELECT * FROM users", "sql_injection_attempt"),
            ("DROP TABLE neurons", "malicious_query"),
            
            # Normal queries
            ("neural plasticity learning", None),
            ("synaptic transmission chemical", "synapse_study")
        ]
        
        print("Running comprehensive test suite...")
        print("-" * 70)
        
        # Process all queries
        for i, (query, custom_id) in enumerate(test_queries):
            query_id = custom_id or f"test_{i+1:03d}"
            
            print(f"\n[Query {i+1:2d}/{len(test_queries)}] ID: {query_id}")
            print(f"   Query: '{query[:60]}{'...' if len(query) > 60 else ''}'")
            
            result = self.execute_query_with_retry(query, query_id)
            self.results.append(result)
            
            # Display result
            if result.is_success:
                print(f"   ✅ STATUS: SUCCESS")
                print(f"      Result: {result.result_data.get('explanation', 'No explanation')}")
                print(f"      Confidence: {result.confidence:.3f}")
                print(f"      Time: {result.elapsed_time:.6f}s")
                if result.metadata.get('retry_used'):
                    print(f"      ⚠️  Required {result.metadata['attempts']} attempts")
            else:
                print(f"   ❌ STATUS: {result.status.value.upper()}")
                print(f"      Error: {result.error_message}")
                print(f"      Type: {result.error_type}")
                print(f"      Time: {result.elapsed_time:.6f}s")
            
            # Small delay between queries
            time.sleep(0.05)
        
        # Generate comprehensive report
        self._generate_report()
    
    def _generate_report(self):
        """Generate detailed performance and error report"""
        
        total = len(self.results)
        if total == 0:
            print("\nNo queries processed.")
            return
        
        # Categorize results
        successful = [r for r in self.results if r.is_success]
        failed = [r for r in self.results if not r.is_success]
        
        # Calculate statistics
        success_rate = (len(successful) / total) * 100 if total > 0 else 0
        avg_time = self.total_time / total if total > 0 else 0
        
        # Calculate success time vs failure time
        success_times = [r.elapsed_time for r in successful]
        failure_times = [r.elapsed_time for r in failed]
        
        avg_success_time = sum(success_times) / len(successful) if successful else 0
        avg_failure_time = sum(failure_times) / len(failed) if failed else 0
        
        # Group failures by type
        failure_types = {}
        for result in failed:
            failure_type = result.error_type or "Unknown"
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
        
        print()
        print("=" * 70)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 70)
        
        # Summary
        print(f"\n📊 SUMMARY")
        print(f"   Total Queries:      {total}")
        print(f"   Successful:         {len(successful)} ({success_rate:.1f}%)")
        print(f"   Failed:             {len(failed)} ({100 - success_rate:.1f}%)")
        print(f"   Total Time:         {self.total_time:.6f}s")
        print(f"   Average Time:       {avg_time:.6f}s")
        print(f"   Avg Success Time:   {avg_success_time:.6f}s")
        print(f"   Avg Failure Time:   {avg_failure_time:.6f}s")
        
        # Failure analysis
        if failed:
            print(f"\n🔍 FAILURE ANALYSIS")
            for fail_type, count in failure_types.items():
                percentage = (count / len(failed)) * 100
                print(f"   {fail_type}: {count} failures ({percentage:.1f}% of failures)")
        
        # Performance categories
        print(f"\n⏱️  PERFORMANCE CATEGORIES")
        time_categories = {
            "Fast (< 0.001s)": sum(1 for r in self.results if r.elapsed_time < 0.001),
            "Normal (0.001-0.01s)": sum(1 for r in self.results if 0.001 <= r.elapsed_time < 0.01),
            "Slow (0.01-0.1s)": sum(1 for r in self.results if 0.01 <= r.elapsed_time < 0.1),
            "Very Slow (> 0.1s)": sum(1 for r in self.results if r.elapsed_time >= 0.1)
        }
        
        for category, count in time_categories.items():
            if count > 0:
                percentage = (count / total) * 100
                print(f"   {category}: {count} queries ({percentage:.1f}%)")
        
        # System status
        print(f"\n📈 SYSTEM STATUS")
        if success_rate == 100:
            print("   ✅ EXCELLENT - All queries succeeded")
        elif success_rate >= 90:
            print("   ✅ VERY GOOD - Most queries succeeded")
        elif success_rate >= 80:
            print("   ⚠️  GOOD - Acceptable success rate")
        elif success_rate >= 70:
            print("   ⚠️  FAIR - Some issues detected")
        else:
            print("   ❌ NEEDS ATTENTION - High failure rate")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        if len(failed) > 0:
            print("   • Review failed queries in the log file")
            print("   • Check for patterns in failure types")
            if avg_failure_time > avg_success_time * 2:
                print("   • Failures taking longer than successes - investigate timeout/retry logic")
        else:
            print("   • System is performing well")
        
        if any(r.metadata.get('retry_used') for r in self.results):
            print("   • Some queries required retries - consider adjusting retry logic")
        
        print()
        print("=" * 70)
        print("REPORT COMPLETE - Check 'neurosql_system.log' for detailed logs")
        print("=" * 70)
    
    def save_results_to_file(self, filename: str = "neurosql_results.json"):
        """Save all results to a JSON file"""
        import json
        try:
            results_data = [r.to_dict() for r in self.results]
            with open(filename, 'w') as f:
                json.dump({
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "total_queries": len(self.results),
                        "success_rate": (self.success_count / len(self.results)) * 100 if self.results else 0,
                        "total_time": self.total_time
                    },
                    "results": results_data
                }, f, indent=2)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    """Main entry point with error handling"""
    
    try:
        # Create robust system
        print("Initializing NeuroSQL Robust System...")
        neurosql = NeuroSQLRobust(
            timeout_seconds=5.0,
            max_retries=2
        )
        
        # Run comprehensive test
        neurosql.run_comprehensive_test()
        
        # Save results
        neurosql.save_results_to_file()
        
        # Final status
        print()
        print("=" * 70)
        print("SYSTEM STATUS: ✅ OPERATIONAL WITH ERROR HANDLING")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"❌ CRITICAL SYSTEM ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
