# web_interface_enhanced.py
"""
Enhanced web interface for NeuroSQL with additional features:
- Real-time graph updates
- Search and filtering
- Graph analytics dashboard
- Export options
- User management (basic)
"""

from flask import Flask, render_template, request, jsonify, session, send_file
from flask_cors import CORS
import json
import io
import networkx as nx
from datetime import datetime
from neurosql_core import NeuroSQL, Concept, WeightedRelationship, RelationshipType
from relationship_retriever import RelationshipRetriever
from performance_optimizer import optimize_neurosql, QueryCache

app = Flask(__name__)
app.secret_key = 'neurosql-secret-key-2024'
CORS(app)

# Initialize with example data
neurosql = None
retriever = None
cache = QueryCache(max_size=512)

def init_neurosql():
    """Initialize NeuroSQL with example data"""
    global neurosql, retriever
    
    try:
        # Try to load from saved file
        neurosql = NeuroSQL("WebGraph")
        neurosql.load_from_file("knowledge_graph.json")
        print("Loaded existing knowledge graph")
    except:
        # Create new example
        from example import create_extended_example
        neurosql = create_extended_example()
        print("Created new example graph")
    
    # Optimize for performance
    optimize_neurosql(neurosql, enable_caching=True, build_index=True)
    
    # Create retriever
    retriever = RelationshipRetriever(neurosql)
    
    return neurosql, retriever

# Initialize on startup
neurosql, retriever = init_neurosql()

@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('enhanced_index.html')

@app.route('/api/stats')
def get_stats():
    """Get graph statistics"""
    stats = {
        "concepts": len(neurosql.concepts),
        "relationships": len(neurosql.relationships),
        "domains": len(set(c.domain for c in neurosql.concepts.values())),
        "abstraction_levels": len(set(c.abstraction_level for c in neurosql.concepts.values())),
        "cache_hit_rate": cache.stats()["hit_rate"] if hasattr(cache, 'stats') else 0,
        "last_updated": datetime.now().isoformat()
    }
    return jsonify(stats)

@app.route('/api/search', methods=['GET'])
def search_concepts():
    """Search concepts by name, domain, or attributes"""
    query = request.args.get('q', '').lower()
    domain = request.args.get('domain', '')
    min_level = request.args.get('min_level', type=int)
    max_level = request.args.get('max_level', type=int)
    
    results = []
    for name, concept in neurosql.concepts.items():
        # Apply filters
        if domain and concept.domain != domain:
            continue
        
        if min_level is not None and concept.abstraction_level < min_level:
            continue
            
        if max_level is not None and concept.abstraction_level > max_level:
            continue
        
        # Search in name
        if query in name.lower():
            results.append({
                'name': name,
                'domain': concept.domain,
                'level': concept.abstraction_level,
                'attributes': concept.attributes
            })
            continue
        
        # Search in attributes
        attribute_match = False
        for key, value in concept.attributes.items():
            if isinstance(value, str) and query in value.lower():
                attribute_match = True
                break
        
        if attribute_match:
            results.append({
                'name': name,
                'domain': concept.domain,
                'level': concept.abstraction_level,
                'attributes': concept.attributes
            })
    
    return jsonify({'results': results[:100]})  # Limit to 100 results

@app.route('/api/analytics/centrality')
def get_centrality_analytics():
    """Get centrality analytics"""
    centrality = retriever.calculate_centrality()
    
    # Get top 10 most central nodes
    top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return jsonify({
        'top_central': [{'concept': k, 'score': v} for k, v in top_central],
        'average_centrality': sum(centrality.values()) / len(centrality) if centrality else 0
    })

@app.route('/api/analytics/communities')
def get_community_analytics():
    """Get community detection results"""
    communities = retriever.find_communities()
    
    if not communities:
        return jsonify({'error': 'Community detection not available'})
    
    # Count community sizes
    community_counts = {}
    for concept, community_id in communities.items():
        community_counts[community_id] = community_counts.get(community_id, 0) + 1
    
    # Get largest communities
    largest_communities = sorted(community_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return jsonify({
        'total_communities': len(set(communities.values())),
        'largest_communities': [{'id': k, 'size': v} for k, v in largest_communities],
        'community_distribution': community_counts
    })

@app.route('/api/graph/export', methods=['GET'])
def export_graph():
    """Export graph in various formats"""
    format_type = request.args.get('format', 'json')
    
    if format_type == 'json':
        # Export as JSON
        neurosql.save_to_file("export_temp.json")
        return send_file("export_temp.json", as_attachment=True, download_name='neurosql_graph.json')
    
    elif format_type == 'csv':
        # Export as CSV
        import csv
        from io import StringIO
        
        si = StringIO()
        writer = csv.writer(si)
        
        # Write concepts
        writer.writerow(['type', 'id', 'name', 'domain', 'level', 'attributes'])
        for name, concept in neurosql.concepts.items():
            writer.writerow(['concept', name, name, concept.domain, 
                           concept.abstraction_level, json.dumps(concept.attributes)])
        
        # Write relationships
        for rel in neurosql.relationships:
            writer.writerow(['relationship', f"{rel.concept_from}_{rel.concept_to}", 
                           rel.concept_from, rel.concept_to, 
                           rel.relationship_type.value, rel.weight])
        
        output = io.BytesIO()
        output.write(si.getvalue().encode('utf-8'))
        output.seek(0)
        
        return send_file(output, as_attachment=True, download_name='neurosql_graph.csv')
    
    elif format_type == 'graphml':
        # Export as GraphML
        graphml_content = '\n'.join(nx.generate_graphml(retriever.graph))
        output = io.BytesIO(graphml_content.encode('utf-8'))
        output.seek(0)
        
        return send_file(output, as_attachment=True, download_name='neurosql_graph.graphml')
    
    else:
        return jsonify({'error': 'Unsupported format'})

@app.route('/api/graph/import', methods=['POST'])
def import_graph():
    """Import graph data"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    format_type = request.args.get('format', 'json')
    
    try:
        if format_type == 'json' and file.filename.endswith('.json'):
            # Save and load
            file.save('import_temp.json')
            neurosql.load_from_file('import_temp.json')
            
        elif format_type == 'csv' and file.filename.endswith('.csv'):
            # Import from CSV
            from data_importer import DataImporter
            importer = DataImporter(neurosql)
            file.save('import_temp.csv')
            importer.import_from_csv('import_temp.csv')
        
        # Rebuild retriever with new data
        global retriever
        retriever = RelationshipRetriever(neurosql)
        
        return jsonify({'success': True, 'message': f'Imported {len(neurosql.concepts)} concepts'})
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/concept/suggest', methods=['GET'])
def suggest_relationships():
    """Suggest potential relationships based on similarity"""
    concept_name = request.args.get('concept')
    if not concept_name:
        return jsonify({'error': 'Concept name required'})
    
    # Simple similarity based on shared attributes and domains
    concept = neurosql.get_concept(concept_name)
    if not concept:
        return jsonify({'error': 'Concept not found'})
    
    suggestions = []
    for other_name, other_concept in neurosql.concepts.items():
        if other_name == concept_name:
            continue
        
        similarity_score = 0
        
        # Domain similarity
        if concept.domain == other_concept.domain:
            similarity_score += 0.3
        
        # Level similarity (closer levels are more similar)
        level_diff = abs(concept.abstraction_level - other_concept.abstraction_level)
        similarity_score += max(0, 1.0 - level_diff * 0.2)
        
        # Attribute similarity
        common_attrs = set(concept.attributes.keys()) & set(other_concept.attributes.keys())
        if common_attrs:
            similarity_score += len(common_attrs) * 0.1
        
        if similarity_score > 0.3:  # Threshold
            suggestions.append({
                'concept': other_name,
                'similarity': similarity_score,
                'domain': other_concept.domain,
                'level': other_concept.abstraction_level
            })
    
    # Sort by similarity
    suggestions.sort(key=lambda x: x['similarity'], reverse=True)
    
    return jsonify({'suggestions': suggestions[:10]})

@app.route('/api/visualization/custom', methods=['POST'])
def custom_visualization():
    """Generate custom visualization with specific settings"""
    data = request.json
    layout = data.get('layout', 'spring')
    highlight = data.get('highlight', [])
    min_weight = data.get('min_weight', 0.0)
    
    # Create a filtered subgraph
    G = retriever.graph
    
    if min_weight > 0:
        # Filter edges by weight
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) 
                          if d.get('weight', 1.0) < min_weight]
        subgraph = G.copy()
        subgraph.remove_edges_from(edges_to_remove)
        # Remove isolated nodes
        isolated_nodes = [n for n in subgraph.nodes() if subgraph.degree(n) == 0]
        subgraph.remove_nodes_from(isolated_nodes)
    else:
        subgraph = G
    
    # Generate visualization
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    
    plt.figure(figsize=(14, 12))
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(subgraph, k=2, iterations=100)
    elif layout == 'circular':
        pos = nx.circular_layout(subgraph)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(subgraph)
    else:
        pos = nx.spring_layout(subgraph)
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    
    # Calculate node sizes based on degree
    degrees = dict(subgraph.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    for node in subgraph.nodes():
        # Color highlighted nodes
        if node in highlight:
            node_colors.append('red')
        else:
            node_colors.append('skyblue')
        
        # Size based on degree
        size = 300 + (degrees.get(node, 0) / max_degree) * 700
        node_sizes.append(size)
    
    nx.draw_networkx_nodes(subgraph, pos, 
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.8)
    
    # Draw edges with weights
    edge_colors = []
    edge_widths = []
    for u, v, data in subgraph.edges(data=True):
        weight = data.get('weight', 1.0)
        edge_colors.append(weight)
        edge_widths.append(1 + weight * 3)
    
    edges = nx.draw_networkx_edges(
        subgraph, pos,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Blues,
        width=edge_widths,
        alpha=0.6
    )
    
    # Draw labels for highlighted nodes
    labels = {node: node for node in highlight if node in subgraph.nodes()}
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=10, font_weight='bold')
    
    plt.title(f"NeuroSQL Graph - {layout.capitalize()} Layout")
    plt.axis('off')
    plt.colorbar(edges, label='Relationship Weight')
    plt.tight_layout()
    
    # Save to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return jsonify({'image': img_str})

@app.route('/api/performance/monitor')
def performance_monitor():
    """Get performance monitoring data"""
    if hasattr(neurosql, 'performance_monitor'):
        stats = neurosql.performance_monitor.get_stats()
        return jsonify(stats)
    
    if hasattr(neurosql, 'query_cache'):
        cache_stats = neurosql.query_cache.stats()
        return jsonify({'cache': cache_stats})
    
    return jsonify({'message': 'Performance monitoring not enabled'})

if __name__ == '__main__':
    print("Starting enhanced NeuroSQL web interface...")
    print(f"Graph loaded: {len(neurosql.concepts)} concepts, {len(neurosql.relationships)} relationships")
    print("Open http://localhost:5001 in your browser")
    app.run(debug=True, port=5001)