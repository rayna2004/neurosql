# neurosql_consistent.py
"""
NeuroSQL with consistent error handling and timing
"""

import sys
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConsistentNeuroSQL:
    """NeuroSQL with proper consistency"""
    
    def __init__(self):
        self.queries_processed = 0
        self.total_elapsed = 0.0
        self.successful_queries = 0
    
    def process_query(self, query_text: str, query_id: Optional[str] = None) -> Dict[str, Any]:
        """Process query with consistent error handling"""
        
        if query_id is None:
            query_id = f"query_{self.queries_processed + 1:06d}"
        
        start_time = time.perf_counter()
        
        try:
            # Simulate query processing (replace with actual logic)
            logger.info(f"Processing query {query_id}: {query_text}")
            
            # Your actual query logic would go here
            time.sleep(0.001)  # Simulate work
            
            # Generate result
            result = {
                "query": query_text,
                "confidence": 0.95,
                "explanation": f"Processed: {query_text}",
                "timestamp": datetime.now().isoformat()
            }
            
            elapsed = time.perf_counter() - start_time
            
            self.queries_processed += 1
            self.successful_queries += 1
            self.total_elapsed += elapsed
            
            return {
                "id": query_id,
                "ok": True,
                "result": result,
                "error": None,
                "elapsed": elapsed  # Store with full precision
            }
            
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Query {query_id} failed: {error_msg}")
            
            self.queries_processed += 1
            self.total_elapsed += elapsed
            
            return {
                "id": query_id,
                "ok": False,
                "result": None,
                "error": error_msg,  # Specific error, not "Unknown error"
                "elapsed": elapsed
            }
    
    def run_demo(self):
        """Run demo with consistent output"""
        
        print("=" * 60)
        print("NEUROSQL CONSISTENT DEMO")
        print("=" * 60)
        
        queries = [
            "dopamine modulates reward",
            "hippocampus supports memory",
            "glutamate excites neurons",
            "gaba inhibits neuronal firing"
        ]
        
        results = []
        
        for i, query in enumerate(queries):
            query_id = f"inf_{i+1:06d}"
            
            # Process query
            response = self.process_query(query, query_id)
            results.append(response)
            
            # Log consistently
            status = "SUCCESS" if response["ok"] else "FAILED"
            logger.info(f"Query {query_id} {status} in {response['elapsed']:.6f}s")
            
            # Display consistently
            if response["ok"]:
                print(f"  ✅ Query: {query}")
                print(f"     Result: {response['result']['explanation']}")
                print(f"     Time: {response['elapsed']:.6f}s")
            else:
                print(f"  ❌ Query: {query}")
                print(f"     Error: {response['error']}")
                print(f"     Time: {response['elapsed']:.6f}s")
        
        # Calculate metrics
        if results:
            success_count = sum(1 for r in results if r["ok"])
            success_rate = (success_count / len(results)) * 100 if results else 0
            avg_time = self.total_elapsed / len(results) if results else 0
            
            print(f"`n{'='*40}")
            print(f"Total Queries: {len(results)}")
            print(f"Successful: {success_count}")
            print(f"Success Rate: {success_rate:.1f}%")
            print(f"Average Time: {avg_time:.6f}s")
            print(f"Total Time: {self.total_elapsed:.6f}s")
        
        return results

def main():
    """Main entry point"""
    system = ConsistentNeuroSQL()
    system.run_demo()

if __name__ == "__main__":
    main()
