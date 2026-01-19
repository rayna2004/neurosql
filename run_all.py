# run_all.py
"""
Run all NeuroSQL features in sequence.
"""

import sys
import time
import webbrowser
import threading
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def test_core_functionality():
    """Test core NeuroSQL functionality"""
    print("\n1. Testing Core Functionality...")
    try:
        from neurosql_core import NeuroSQL, Concept, WeightedRelationship, RelationshipType
        from example import create_extended_example
        
        neurosql = create_extended_example()
        print(f"   ✓ Created graph with {len(neurosql.concepts)} concepts")
        print(f"   ✓ Graph has {len(neurosql.relationships)} relationships")
        
        # Save example
        neurosql.save_to_file("knowledge_graph.json")
        print(f"   ✓ Saved to 'knowledge_graph.json'")
        
        return neurosql
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return None

def test_graph_algorithms(neurosql):
    """Test graph algorithms"""
    print("\n2. Testing Graph Algorithms...")
    try:
        from relationship_retriever import RelationshipRetriever
        
        retriever = RelationshipRetriever(neurosql)
        
        # Test BFS
        bfs_result = retriever.bfs_traversal("Fluffy", max_depth=2)
        print(f"   ✓ BFS traversal found {len(bfs_result)} concepts")
        
        # Test shortest path
        path = retriever.find_shortest_path("Fluffy", "LivingBeing")
        if path:
            print(f"   ✓ Shortest path: {len(path)} steps ({' → '.join(path)})")
        else:
            print(f"   ✓ No path found (expected for some graphs)")
        
        # Test centrality
        centrality = retriever.calculate_centrality()
        print(f"   ✓ Calculated centrality for {len(centrality)} nodes")
        
        return retriever
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return None

def test_data_operations(neurosql):
    """Test data import/export operations"""
    print("\n3. Testing Data Operations...")
    try:
        # Export
        neurosql.save_to_file("test_export.json")
        print(f"   ✓ Exported graph to 'test_export.json'")
        
        # Import test
        new_neurosql = NeuroSQL("TestImport")
        new_neurosql.load_from_file("test_export.json")
        print(f"   ✓ Imported graph with {len(new_neurosql.concepts)} concepts")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def generate_visualization(retriever):
    """Generate graph visualization"""
    print("\n4. Generating Visualizations...")
    try:
        retriever.visualize_graph("neurosql_demo_graph.png")
        print(f"   ✓ Generated 'neurosql_demo_graph.png'")
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def run_web_interface(port=5000, enhanced=False):
    """Run web interface"""
    print(f"\nStarting web interface on port {port}...")
    print(f"Open http://localhost:{port} in your browser")
    
    if enhanced:
        from web_interface_enhanced import app
    else:
        from web_interface import app
    
    import threading
    
    def run_app():
        app.run(debug=True, port=port, use_reloader=False)
    
    thread = threading.Thread(target=run_app)
    thread.daemon = True
    thread.start()
    
    time.sleep(2)  # Give server time to start
    webbrowser.open(f"http://localhost:{port}")
    
    return thread

def main():
    """Run all NeuroSQL features"""
    
    print_header("NEUROSQL COMPLETE DEMONSTRATION")
    
    # Run core tests
    neurosql = test_core_functionality()
    if not neurosql:
        return
    
    retriever = test_graph_algorithms(neurosql)
    if not retriever:
        return
    
    test_data_operations(neurosql)
    generate_visualization(retriever)
    
    # Ask user what they want to do
    print("\n" + "="*60)
    print("WHAT WOULD YOU LIKE TO DO NEXT?")
    print("="*60)
    print("\nOptions:")
    print("  1. Start basic web interface (port 5000)")
    print("  2. Start enhanced web interface (port 5001)")
    print("  3. Run performance tests")
    print("  4. Run all unit tests")
    print("  5. Exit")
    
    while True:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            # Run basic web interface
            thread = run_web_interface(5000, enhanced=False)
            
            try:
                print("\nWeb interface running. Press Ctrl+C to stop.")
                while thread.is_alive():
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
                
        elif choice == "2":
            # Try to run enhanced interface
            try:
                import web_interface_enhanced
                thread = run_web_interface(5001, enhanced=True)
                
                try:
                    print("\nEnhanced web interface running. Press Ctrl+C to stop.")
                    while thread.is_alive():
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nShutting down...")
                    break
                    
            except ImportError as e:
                print(f"\nEnhanced interface not available: {e}")
                print("You may need to install additional dependencies.")
                print("Try: pip install flask-cors flask-bootstrap")
                
        elif choice == "3":
            # Run performance tests
            print("\nRunning performance tests...")
            try:
                from large_dataset_generator import generate_large_dataset, performance_test
                
                print("Generating test dataset (100 concepts, 200 relationships)...")
                test_neurosql = generate_large_dataset(100, 200)
                performance_test(test_neurosql)
                
            except Exception as e:
                print(f"Error running performance tests: {e}")
                
        elif choice == "4":
            # Run unit tests
            print("\nRunning unit tests...")
            try:
                import unittest
                import os
                
                # Change to tests directory
                original_dir = os.getcwd()
                tests_dir = Path(__file__).parent / "tests"
                
                if tests_dir.exists():
                    os.chdir(tests_dir)
                    
                    # Discover and run tests
                    test_suite = unittest.defaultTestLoader.discover('.', pattern='test_*.py')
                    runner = unittest.TextTestRunner(verbosity=2)
                    result = runner.run(test_suite)
                    
                    os.chdir(original_dir)
                    
                    if result.wasSuccessful():
                        print("\n✓ All tests passed!")
                    else:
                        print("\n✗ Some tests failed.")
                else:
                    print("Tests directory not found.")
                    
            except Exception as e:
                print(f"Error running tests: {e}")
                
        elif choice == "5":
            print("\nExiting. You can run individual components:")
            print("  python example.py")
            print("  python web_interface.py")
            print("  python large_dataset_generator.py")
            break
            
        else:
            print("Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()