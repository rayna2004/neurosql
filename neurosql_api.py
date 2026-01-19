# neurosql_api.py
"""
RESTful API for NeuroSQL with advanced features.
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
from typing import Dict, Any

from neurosql_advanced import NeuroSQLAdvanced, ReasoningOperator

app = Flask(__name__)
CORS(app)

# Global NeuroSQL instance
neurosql = NeuroSQLAdvanced("APIDemo")

# Initialize with example data
from example import create_extended_example
example_neurosql = create_extended_example()

# Copy data to our advanced instance
for name, concept in example_neurosql.concepts.items():
    neurosql.neurosql.concepts[name] = concept
for rel in example_neurosql.relationships:
    neurosql.neurosql.relationships.append(rel)

@app.route('/')
def index():
    return jsonify({
        'name': 'NeuroSQL API',
        'version': '2.0.0',
        'features': [
            'semantic_knowledge_graph',
            'declarative_query_language',
            'advanced_reasoning',
            'restful_api'
        ]
    })

@app.route('/api/concepts', methods=['GET'])
def get_concepts():
    """Get all concepts with optional filtering"""
    domain = request.args.get('domain')
    min_level = request.args.get('min_level', type=int)
    max_level = request.args.get('max_level', type=int)
    
    concepts = []
    for name, concept in neurosql.neurosql.concepts.items():
        if domain and concept.domain != domain:
            continue
        if min_level is not None and concept.abstraction_level < min_level:
            continue
        if max_level is not None and concept.abstraction_level > max_level:
            continue
        
        concepts.append({
            'name': name,
            'domain': concept.domain,
            'abstraction_level': concept.abstraction_level,
            'attributes': concept.attributes
        })
    
    return jsonify({'concepts': concepts})

@app.route('/api/concepts', methods=['POST'])
def create_concept():
    """Create a new concept"""
    data = request.json
    
    if not data or 'name' not in data:
        return jsonify({'error': 'Concept name required'}), 400
    
    try:
        from neurosql_core import Concept
        concept = Concept(
            name=data['name'],
            attributes=data.get('attributes', {}),
            abstraction_level=data.get('abstraction_level', 0),
            domain=data.get('domain', 'general')
        )
        
        neurosql.neurosql.add_concept(concept)
        
        return jsonify({
            'message': 'Concept created successfully',
            'concept': data['name']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/query', methods=['POST'])
def execute_query():
    """Execute a declarative query"""
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({'error': 'Query required'}), 400
    
    try:
        result = neurosql.execute(data['query'])
        return jsonify({
            'result': result,
            'count': len(result)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/reason', methods=['POST'])
def apply_reasoning():
    """Apply reasoning operators"""
    data = request.json
    
    if not data or 'operator' not in data:
        return jsonify({'error': 'Operator required'}), 400
    
    try:
        operator = ReasoningOperator(data['operator'])
        params = data.get('params', {})
        
        result = neurosql.reason(operator, params)
        
        return jsonify({
            'operator': operator.value,
            'inferred_relationships': len(result.relationships),
            'confidence': result.confidence,
            'explanation': result.explain()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/export', methods=['GET'])
def export_knowledge():
    """Export knowledge in various formats"""
    format_type = request.args.get('format', 'json-ld')
    
    try:
        if format_type == 'json-ld':
            content = neurosql.export_knowledge('json-ld')
            return Response(
                content,
                mimetype='application/ld+json',
                headers={'Content-Disposition': 'attachment;filename=knowledge.jsonld'}
            )
        elif format_type == 'rdf':
            content = neurosql.export_knowledge('rdf')
            return Response(
                content,
                mimetype='application/rdf+xml',
                headers={'Content-Disposition': 'attachment;filename=knowledge.rdf'}
            )
        else:
            return jsonify({'error': 'Unsupported format'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/validate', methods=['GET'])
def validate_knowledge():
    """Validate the knowledge base"""
    issues = neurosql.validate_knowledge_base()
    
    return jsonify({
        'issues': issues,
        'valid': all(len(v) == 0 for v in issues.values())
    })

@app.route('/api/explain/<path:relationship_id>', methods=['GET'])
def explain_relationship(relationship_id):
    """Explain a relationship"""
    # Parse relationship ID (simplified)
    parts = relationship_id.split('->')
    if len(parts) != 2:
        return jsonify({'error': 'Invalid relationship format'}), 400
    
    from_concept, to_concept = parts
    
    # Find the relationship
    relationships = neurosql.neurosql.find_relationships(from_concept)
    target_rel = None
    
    for rel in relationships:
        if rel.concept_to == to_concept:
            target_rel = rel
            break
    
    if not target_rel:
        return jsonify({'error': 'Relationship not found'}), 404
    
    explanation = neurosql.explain(target_rel)
    
    return jsonify({
        'relationship': f"{from_concept} -> {to_concept}",
        'explanation': explanation,
        'type': target_rel.relationship_type.value,
        'weight': target_rel.weight
    })

@app.route('/api/operators', methods=['GET'])
def list_operators():
    """List available reasoning operators"""
    operators = [
        {
            'name': op.value,
            'description': _get_operator_description(op)
        }
        for op in ReasoningOperator
    ]
    
    return jsonify({'operators': operators})

def _get_operator_description(operator: ReasoningOperator) -> str:
    """Get description for a reasoning operator"""
    descriptions = {
        ReasoningOperator.TRANSITIVE_CLOSURE: 
            "Compute transitive closure for relationships (e.g., if A is_a B and B is_a C, infer A is_a C)",
        ReasoningOperator.PROPERTY_INHERITANCE:
            "Inherit properties along IS_A hierarchy",
        ReasoningOperator.CONFIDENCE_PROPAGATION:
            "Propagate confidence scores through the graph",
        ReasoningOperator.COMMON_DESCENDANT:
            "Infer relationships based on common descendants",
        ReasoningOperator.COMMON_ANCESTOR:
            "Infer relationships based on common ancestors",
        ReasoningOperator.SIMILARITY_INFERENCE:
            "Infer relationships based on attribute similarity",
        ReasoningOperator.ANALOGICAL_REASONING:
            "Apply analogical reasoning (A:B::C:D)",
        ReasoningOperator.DEFAULT_REASONING:
            "Apply domain-specific default reasoning"
    }
    
    return descriptions.get(operator, "No description available")

if __name__ == '__main__':
    print("Starting NeuroSQL Advanced API...")
    print(f"Loaded {len(neurosql.neurosql.concepts)} concepts")
    print(f"Loaded {len(neurosql.neurosql.relationships)} relationships")
    print("API available at: http://localhost:8000")
    app.run(debug=True, port=8000)