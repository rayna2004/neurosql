import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print('Testing NeuroSQL setup...')
print('=' * 50)

try:
    # Test imports
    from neurosql_core import NeuroSQL, Concept
    print('✓ neurosql_core imported')
    
    # Test Flask
    from flask import Flask
    print('✓ Flask imported')
    
    # Check templates folder
    if os.path.exists('templates'):
        print('✓ templates folder exists')
        if os.path.exists('templates/index.html'):
            print('✓ templates/index.html exists')
        else:
            print('⚠ templates/index.html not found')
    else:
        print('✗ templates folder missing')
    
    # Test creating a simple graph
    neurosql = NeuroSQL('Test')
    concept = Concept('Test Concept', {}, 0, 'test')
    neurosql.add_concept(concept)
    print(f'✓ Created test graph with {len(neurosql.concepts)} concepts')
    
    print('\n' + '=' * 50)
    print('✅ Setup looks good!')
    
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
