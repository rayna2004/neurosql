# web_interface.py
"""
Basic web interface for NeuroSQL.
"""

from flask import Flask, render_template, request, jsonify
import json
from neurosql_core import NeuroSQL, Concept, WeightedRelationship, RelationshipType
from relationship_retriever import RelationshipRetriever

app = Flask(__name__)

# Initialize NeuroSQL instance
try:
    # Try to load existing graph
    neurosql = NeuroSQL("Web Interface Graph")
    neurosql.load_from_file("knowledge_graph.json")
    print(f"Loaded existing graph with {len(neurosql.concepts)} concepts")
except FileNotFoundError:
    # Create example if no saved file
    from example import create_extended_example
    neurosql = create_extended_example()
    print(f"Created new example graph with {len(neurosql.concepts)} concepts")

retriever = RelationshipRetriever(neurosql)

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/concepts', methods=['GET'])
def get_concepts():
    """Get all concepts"""
    concepts = []
    for name, concept in neurosql.concepts.items():
        concepts.append({
            'name': name,
            'attributes': concept.attributes,
            'abstraction_level': concept.abstraction_level,
            'domain': concept.domain
        })
    return jsonify({'concepts': concepts})

@app.route('/api/concepts', methods=['POST'])
def add_concept():
    """Add a new concept"""
    try:
        data = request.json
        if not data or 'name' not in data:
            return jsonify({'error': 'Concept name required'}), 400
        
        concept = Concept(
            name=data['name'],
            attributes=data.get('attributes', {}),
            abstraction_level=data.get('abstraction_level', 0),
            domain=data.get('domain', 'general')
        )
        neurosql.add_concept(concept)
        
        # Rebuild retriever with new concept
        global retriever
        retriever = RelationshipRetriever(neurosql)
        
        return jsonify({
            'message': 'Concept added successfully',
            'concept': data['name']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/relationships', methods=['GET'])
def get_relationships():
    """Get relationships for a concept"""
    concept_name = request.args.get('concept')
    if not concept_name:
        # Return all relationships if no concept specified
        relationships = []
        for rel in neurosql.relationships:
            relationships.append({
                'from': rel.concept_from,
                'to': rel.concept_to,
                'type': rel.relationship_type.value,
                'weight': rel.weight
            })
        return jsonify({'relationships': relationships})
    
    relationships = neurosql.find_relationships(concept_name)
    rel_data = []
    for rel in relationships:
        rel_data.append({
            'from': rel.concept_from,
            'to': rel.concept_to,
            'type': rel.relationship_type.value,
            'weight': rel.weight
        })
    return jsonify({'relationships': rel_data})

@app.route('/api/relationships', methods=['POST'])
def add_relationship():
    """Add a new relationship"""
    try:
        data = request.json
        required_fields = ['from', 'to', 'type']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate relationship type
        try:
            rel_type = RelationshipType(data['type'])
        except ValueError:
            return jsonify({'error': f'Invalid relationship type: {data["type"]}'}), 400
        
        # Check if concepts exist
        if data['from'] not in neurosql.concepts:
            return jsonify({'error': f'Source concept not found: {data["from"]}'}), 400
        if data['to'] not in neurosql.concepts:
            return jsonify({'error': f'Target concept not found: {data["to"]}'}), 400
        
        relationship = WeightedRelationship(
            concept_from=data['from'],
            concept_to=data['to'],
            relationship_type=rel_type,
            weight=data.get('weight', 1.0)
        )
        neurosql.add_weighted_relationship(relationship)
        
        # Rebuild retriever with new relationship
        global retriever
        retriever = RelationshipRetriever(neurosql)
        
        return jsonify({
            'message': 'Relationship added successfully',
            'relationship': {
                'from': data['from'],
                'to': data['to'],
                'type': data['type']
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/traverse/bfs', methods=['GET'])
def bfs_traverse():
    """Perform BFS traversal"""
    start = request.args.get('start')
    depth = int(request.args.get('depth', 3))
    
    if not start:
        return jsonify({'error': 'Start concept required'}), 400
    
    if start not in neurosql.concepts:
        return jsonify({'error': f'Concept not found: {start}'}), 400
    
    try:
        traversal = retriever.bfs_traversal(start, max_depth=depth)
        return jsonify({'traversal': traversal})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/traverse/dfs', methods=['GET'])
def dfs_traverse():
    """Perform DFS traversal"""
    start = request.args.get('start')
    depth = int(request.args.get('depth', 3))
    
    if not start:
        return jsonify({'error': 'Start concept required'}), 400
    
    if start not in neurosql.concepts:
        return jsonify({'error': f'Concept not found: {start}'}), 400
    
    try:
        traversal = retriever.dfs_traversal(start, max_depth=depth)
        return jsonify({'traversal': traversal})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/shortest-path', methods=['GET'])
def shortest_path():
    """Find shortest path between two concepts"""
    start = request.args.get('start')
    end = request.args.get('end')
    
    if not start or not end:
        return jsonify({'error': 'Both start and end concepts required'}), 400
    
    if start not in neurosql.concepts:
        return jsonify({'error': f'Start concept not found: {start}'}), 400
    if end not in neurosql.concepts:
        return jsonify({'error': f'End concept not found: {end}'}), 400
    
    try:
        path = retriever.find_shortest_path(start, end)
        return jsonify({'path': path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualize', methods=['GET'])
def visualize():
    """Generate and return graph visualization"""
    try:
        import io
        import base64
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Create visualization
        filename = "temp_graph.png"
        retriever.visualize_graph(filename)
        
        # Convert to base64 for web display
        with open(filename, "rb") as img_file:
            img_str = base64.b64encode(img_file.read()).decode('utf-8')
        
        return jsonify({'image': img_str})
        
    except ImportError:
        return jsonify({'error': 'Matplotlib not available'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get graph statistics"""
    stats = {
        'concepts': len(neurosql.concepts),
        'relationships': len(neurosql.relationships),
        'domains': len(set(c.domain for c in neurosql.concepts.values())),
        'abstraction_levels': len(set(c.abstraction_level for c in neurosql.concepts.values())),
        'relationship_types': len(set(r.relationship_type.value for r in neurosql.relationships))
    }
    return jsonify(stats)

@app.route('/api/save', methods=['POST'])
def save_graph():
    """Save the current graph"""
    try:
        filename = request.json.get('filename', 'neurosql_graph.json')
        neurosql.save_to_file(filename)
        return jsonify({'message': f'Graph saved to {filename}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load', methods=['POST'])
def load_graph():
    """Load a graph from file"""
    try:
        filename = request.json.get('filename', 'neurosql_graph.json')
        neurosql.load_from_file(filename)
        
        # Rebuild retriever with loaded data
        global retriever
        retriever = RelationshipRetriever(neurosql)
        
        return jsonify({
            'message': f'Graph loaded from {filename}',
            'concepts': len(neurosql.concepts),
            'relationships': len(neurosql.relationships)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Starting NeuroSQL Web Interface...")
    print(f"Graph loaded: {len(neurosql.concepts)} concepts, {len(neurosql.relationships)} relationships")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000, use_reloader=False)