import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Optional
from collections import deque
from neurosql_core import NeuroSQL, WeightedRelationship, RelationshipType


class RelationshipRetriever:
    def __init__(self, neurosql: NeuroSQL):
        self.neurosql = neurosql
        self.graph = self._build_graph()
    
    def _build_graph(self) -> nx.DiGraph:
        """Build NetworkX graph from NeuroSQL data"""
        G = nx.DiGraph()
        
        # Add nodes (concepts)
        for name, concept in self.neurosql.concepts.items():
            G.add_node(
                name, 
                attributes=concept.attributes,
                abstraction_level=concept.abstraction_level,
                domain=concept.domain
            )
        
        # Add edges (relationships)
        for rel in self.neurosql.relationships:
            G.add_edge(
                rel.concept_from, 
                rel.concept_to,
                weight=rel.weight,
                relationship_type=rel.relationship_type.value,
                confidence=rel.confidence,
                **rel.metadata
            )
        
        return G
    
    def rebuild_graph(self) -> None:
        """Rebuild the graph after changes"""
        self.graph = self._build_graph()
    
    def bfs_traversal(self, start_concept: str, max_depth: int = 3) -> List[str]:
        """Breadth-First Search traversal"""
        if start_concept not in self.graph:
            return []
        
        visited = set()
        queue = deque([(start_concept, 0)])
        traversal_order = []
        
        while queue:
            current, depth = queue.popleft()
            if current not in visited and depth <= max_depth:
                visited.add(current)
                traversal_order.append(current)
                
                # Get neighbors (both successors and predecessors)
                neighbors = list(self.graph.successors(current)) + list(self.graph.predecessors(current))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
        
        return traversal_order
    
    def dfs_traversal(self, start_concept: str, max_depth: int = 3) -> List[str]:
        """Depth-First Search traversal"""
        if start_concept not in self.graph:
            return []
        
        visited = set()
        traversal_order = []
        
        def dfs_recursive(node: str, depth: int):
            if depth > max_depth or node in visited:
                return
            
            visited.add(node)
            traversal_order.append(node)
            
            # Explore neighbors
            neighbors = list(self.graph.successors(node)) + list(self.graph.predecessors(node))
            for neighbor in neighbors:
                dfs_recursive(neighbor, depth + 1)
        
        dfs_recursive(start_concept, 0)
        return traversal_order
    
    def find_shortest_path(self, start: str, end: str) -> List[str]:
        """Find shortest path using Dijkstra's algorithm (weighted)"""
        try:
            return nx.dijkstra_path(self.graph, start, end, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def find_all_paths(self, start: str, end: str, max_length: int = 5) -> List[List[str]]:
        """Find all paths between two concepts up to max_length"""
        try:
            return list(nx.all_simple_paths(self.graph, start, end, cutoff=max_length))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def find_strongly_connected_components(self) -> List[Set[str]]:
        """Find strongly connected components"""
        return list(nx.strongly_connected_components(self.graph))
    
    def find_weakly_connected_components(self) -> List[Set[str]]:
        """Find weakly connected components"""
        return list(nx.weakly_connected_components(self.graph))
    
    def calculate_centrality(self) -> Dict[str, float]:
        """Calculate betweenness centrality for nodes"""
        return nx.betweenness_centrality(self.graph, weight='weight')
    
    def calculate_pagerank(self) -> Dict[str, float]:
        """Calculate PageRank for nodes"""
        return nx.pagerank(self.graph, weight='weight')
    
    def get_node_degree(self, concept_name: str) -> Dict[str, int]:
        """Get in-degree and out-degree for a node"""
        if concept_name not in self.graph:
            return {"in_degree": 0, "out_degree": 0, "total": 0}
        
        return {
            "in_degree": self.graph.in_degree(concept_name),
            "out_degree": self.graph.out_degree(concept_name),
            "total": self.graph.degree(concept_name)
        }
    
    def find_communities(self) -> Dict[str, int]:
        """Find communities using Louvain method"""
        try:
            import community as community_louvain
            undirected_graph = self.graph.to_undirected()
            partition = community_louvain.best_partition(undirected_graph)
            return partition
        except ImportError:
            print("Install python-louvain: pip install python-louvain")
            return {}
    
    def visualize_graph(self, filename: str = "graph.png", highlight_nodes: List[str] = None):
        """Visualize the graph using NetworkX and Matplotlib"""
        if len(self.graph.nodes()) == 0:
            print("Graph is empty, nothing to visualize")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        
        # Determine node colors
        node_colors = []
        for node in self.graph.nodes():
            if highlight_nodes and node in highlight_nodes:
                node_colors.append('red')
            else:
                node_colors.append('skyblue')
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos, 
            node_color=node_colors, 
            node_size=700, 
            alpha=0.8
        )
        
        # Draw edges
        edge_weights = [self.graph[u][v].get('weight', 1.0) for u, v in self.graph.edges()]
        
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color='gray',
            width=[w * 2 for w in edge_weights],
            alpha=0.6,
            arrows=True,
            arrowsize=20
        )
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=9, font_weight='bold')
        
        # Draw edge labels
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            rel_type = data.get('relationship_type', '')
            weight = data.get('weight', 1.0)
            edge_labels[(u, v)] = f"{rel_type}\n({weight:.2f})"
        
        nx.draw_networkx_edge_labels(
            self.graph, pos, 
            edge_labels=edge_labels, 
            font_size=7,
            alpha=0.8
        )
        
        plt.title(f"NeuroSQL Graph: {self.neurosql.name}", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Graph saved to {filename}")
    
    def get_subgraph(self, concept_names: List[str]) -> nx.DiGraph:
        """Extract a subgraph containing only the specified concepts"""
        return self.graph.subgraph(concept_names).copy()
    
    def get_neighbors(self, concept_name: str, direction: str = "both") -> List[str]:
        """Get neighbors of a concept
        
        Args:
            concept_name: Name of the concept
            direction: "in", "out", or "both"
        """
        if concept_name not in self.graph:
            return []
        
        if direction == "in":
            return list(self.graph.predecessors(concept_name))
        elif direction == "out":
            return list(self.graph.successors(concept_name))
        else:
            return list(set(
                list(self.graph.predecessors(concept_name)) + 
                list(self.graph.successors(concept_name))
            ))