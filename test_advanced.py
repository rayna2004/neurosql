# test_advanced.py
print("Testing NeuroSQL Advanced Features...")
print("=" * 50)

try:
    import numpy as np
    print(f"✓ numpy {np.__version__}")
    
    import rdflib
    print(f"✓ rdflib {rdflib.__version__}")
    
    import jsonschema
    print(f"✓ jsonschema {jsonschema.__version__}")
    
    import pydantic
    print(f"✓ pydantic {pydantic.__version__}")
    
    from flask_cors import CORS
    print("✓ flask-cors")
    
    print("\n✅ All advanced packages installed!")
    
except ImportError as e:
    print(f"\n❌ Missing package: {e}")
    print("\nInstall with: pip install numpy rdflib jsonschema pydantic flask-cors")
