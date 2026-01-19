# neurosql_advanced.py
from neurosql_core import NeuroSQL, Concept, WeightedRelationship, RelationshipType
from query_engine import NeuroSQLQueryLanguage

class ReasoningResult:
    def __init__(self):
        self.relationships = []
        self.steps = []
        self.count = 0

class SimpleReasoningEngine:
    def __init__(self, neurosql):
        self.neurosql = neurosql
    
    def transitive_closure(self, rel_type=None):
        if rel_type is None:
            rel_type = RelationshipType.IS_A
        result = ReasoningResult()
        edges = {}
        for rel in self.neurosql.relationships:
            if rel.relationship_type == rel_type:
                if rel.concept_from not in edges:
                    edges[rel.concept_from] = []
                edges[rel.concept_from].append((rel.concept_to, rel.weight))
        
        for start in edges:
            visited = set()
            stack = [(start, 1.0, [start])]
            while stack:
                current, weight, path = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                if current in edges:
                    for next_node, edge_weight in edges[current]:
                        if next_node not in visited:
                            new_weight = weight * edge_weight
                            new_path = path + [next_node]
                            if len(new_path) > 2:
                                exists = any(r.concept_from == start and r.concept_to == next_node for r in self.neurosql.relationships)
                                if not exists:
                                    new_rel = WeightedRelationship(start, next_node, rel_type, new_weight, metadata={"inferred": True})
                                    result.relationships.append(new_rel)
                                    result.steps.append(f"{start} -> {next_node}")
                                    result.count += 1
                            stack.append((next_node, new_weight, new_path))
        return result

class NeuroSQLAdvanced:
    def __init__(self, name="neurosql_advanced"):
        self.neurosql = NeuroSQL(name)
        self.query_engine = NeuroSQLQueryLanguage(self.neurosql)
        self.reasoning = SimpleReasoningEngine(self.neurosql)
    
    def query(self, q):
        return self.query_engine.execute(q)
    
    def add_concept(self, c):
        self.neurosql.add_concept(c)
    
    def add_relationship(self, r):
        self.neurosql.add_weighted_relationship(r)
    
    def reason_transitive(self):
        result = self.reasoning.transitive_closure()
        for rel in result.relationships:
            self.neurosql.add_weighted_relationship(rel)
        return result
    
    def demo(self):
        print("=" * 60)
        print("NEUROSQL ADVANCED DEMO")
        print("=" * 60)
        
        for c in [Concept("Python", {}, 1, "cs"), Concept("ProgrammingLanguage", {}, 2, "cs"), Concept("Software", {}, 3, "cs")]:
            self.add_concept(c)
        
        self.add_relationship(WeightedRelationship("Python", "ProgrammingLanguage", RelationshipType.IS_A, 0.98))
        self.add_relationship(WeightedRelationship("ProgrammingLanguage", "Software", RelationshipType.IS_A, 0.99))
        
        print(f"\n1. Created: {len(self.neurosql.concepts)} concepts, {len(self.neurosql.relationships)} relationships")
        
        print("\n2. FIND PATH before reasoning:")
        result = self.query('FIND PATH FROM "Python" TO "Software"')
        print(f"   Path: {' -> '.join(result[0]['path']) if result else 'None'}")
        
        print("\n3. Applying transitive closure...")
        tc = self.reason_transitive()
        print(f"   Inferred {tc.count} relationships")
        
        print("\n4. FIND PATH after reasoning:")
        result = self.query('FIND PATH FROM "Python" TO "Software"')
        print(f"   Path: {' -> '.join(result[0]['path']) if result else 'None'}")
        
        print(f"\n5. Final: {len(self.neurosql.relationships)} relationships")
        print("=" * 60)

if __name__ == "__main__":
    NeuroSQLAdvanced().demo()

############################################################
# KEY INSIGHT: The system needs ontology constraints
# to prevent nonsense inferences like:
# - Fluffy → DeepLearning (instance→different domain)
# - Dog → NeuralNetwork (cross-domain)
# - Animal → DeepLearning (cross-domain)
#
# Without constraints, reasoning produces contamination.
# With constraints, only valid inferences are made.
############################################################
