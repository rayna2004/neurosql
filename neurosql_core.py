import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class RelationshipType(Enum):
    IS_A = "is_a"
    PART_OF = "part_of"
    HAS_PROPERTY = "has_property"
    CAUSES = "causes"
    USES = "uses"
    RELATED_TO = "related_to"


@dataclass
class Concept:
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    abstraction_level: int = 0  # 0=concrete, higher=more abstract
    domain: str = "general"


@dataclass
class WeightedRelationship:
    concept_from: str
    concept_to: str
    relationship_type: RelationshipType
    weight: float = 1.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class NeuroSQL:
    def __init__(self, name: str = "neurosql_graph"):
        self.name = name
        self.concepts: Dict[str, Concept] = {}
        self.relationships: List[WeightedRelationship] = []
        self.layers = {
            "concrete": [],
            "abstract": [],
            "meta": []
        }
    
    def add_concept(self, concept: Concept, layer: str = "concrete") -> None:
        """Add a concept to the graph with specified abstraction layer"""
        self.concepts[concept.name] = concept
        if layer in self.layers:
            self.layers[layer].append(concept.name)
    
    def add_weighted_relationship(self, relationship: WeightedRelationship) -> None:
        """Add a weighted relationship between concepts"""
        self.relationships.append(relationship)
    
    def get_concept(self, name: str) -> Optional[Concept]:
        """Retrieve a concept by name"""
        return self.concepts.get(name)
    
    def find_relationships(self, concept_name: str, 
                          relationship_type: Optional[RelationshipType] = None,
                          min_weight: float = 0.0) -> List[WeightedRelationship]:
        """Find relationships for a concept with optional filters"""
        results = []
        for rel in self.relationships:
            if (rel.concept_from == concept_name or rel.concept_to == concept_name):
                if relationship_type is None or rel.relationship_type == relationship_type:
                    if rel.weight >= min_weight:
                        results.append(rel)
        return results
    
    def get_abstraction_hierarchy(self, concept_name: str) -> List[Concept]:
        """Get concepts at different abstraction levels"""
        concept = self.get_concept(concept_name)
        if not concept:
            return []
        
        related_concepts = []
        for rel in self.relationships:
            if rel.concept_from == concept_name and rel.relationship_type == RelationshipType.IS_A:
                related = self.get_concept(rel.concept_to)
                if related:
                    related_concepts.append(related)
        
        return sorted(related_concepts, key=lambda c: c.abstraction_level)
    
    def save_to_file(self, filename: str) -> None:
        """Save the graph to a JSON file"""
        data = {
            "name": self.name,
            "concepts": {
                name: {
                    "name": concept.name,
                    "attributes": concept.attributes,
                    "abstraction_level": concept.abstraction_level,
                    "domain": concept.domain
                } 
                for name, concept in self.concepts.items()
            },
            "relationships": [
                {
                    "concept_from": rel.concept_from,
                    "concept_to": rel.concept_to,
                    "relationship_type": rel.relationship_type.value,
                    "weight": rel.weight,
                    "confidence": rel.confidence,
                    "metadata": rel.metadata
                }
                for rel in self.relationships
            ],
            "layers": self.layers
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filename: str) -> None:
        """Load the graph from a JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.name = data.get("name", self.name)
        
        self.concepts = {}
        for name, concept_data in data["concepts"].items():
            self.concepts[name] = Concept(
                name=concept_data["name"],
                attributes=concept_data.get("attributes", {}),
                abstraction_level=concept_data.get("abstraction_level", 0),
                domain=concept_data.get("domain", "general")
            )
        
        self.relationships = []
        for rel_data in data["relationships"]:
            self.relationships.append(WeightedRelationship(
                concept_from=rel_data["concept_from"],
                concept_to=rel_data["concept_to"],
                relationship_type=RelationshipType(rel_data["relationship_type"]),
                weight=rel_data.get("weight", 1.0),
                confidence=rel_data.get("confidence", 1.0),
                metadata=rel_data.get("metadata", {})
            ))
        
        self.layers = data.get("layers", self.layers)