# reasoning_engine.py
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class ReasoningOperator(Enum):
    TRANSITIVE_CLOSURE = "transitive_closure"
    PROPERTY_INHERITANCE = "property_inheritance"
    DEFAULT_REASONING = "default_reasoning"

@dataclass
class InferenceResult:
    relationships: List[Any]
    confidence: float
    reasoning_steps: List[str]
    operator: ReasoningOperator
    
    def explain(self) -> str:
        explanation = f"Inferred {len(self.relationships)} relationships using {self.operator.value}"
        if self.reasoning_steps:
            explanation += f"\nSteps:\n" + "\n".join(f"  - {step}" for step in self.reasoning_steps)
        return explanation

class ReasoningEngine:
    def __init__(self, neurosql):
        self.neurosql = neurosql
    
    def apply_operator(self, operator: ReasoningOperator, params: Dict[str, Any] = None):
        params = params or {}
        
        if operator == ReasoningOperator.TRANSITIVE_CLOSURE:
            return self._transitive_closure(params)
        elif operator == ReasoningOperator.PROPERTY_INHERITANCE:
            return self._property_inheritance(params)
        elif operator == ReasoningOperator.DEFAULT_REASONING:
            return self._default_reasoning(params)
        else:
            raise ValueError(f"Unknown operator: {operator}")
    
    def _transitive_closure(self, params: Dict[str, Any]):
        from neurosql_core import RelationshipType, WeightedRelationship
        
        start_concept = params.get('start_concept', '')
        relationship_type = params.get('relationship_type', 'is_a')
        
        new_relationships = []
        reasoning_steps = []
        
        # Simple transitive reasoning
        if start_concept in self.neurosql.concepts:
            # Get direct relationships
            direct_rels = self.neurosql.find_relationships(start_concept, RelationshipType(relationship_type))
            
            for rel in direct_rels:
                # Get relationships of the target
                indirect_rels = self.neurosql.find_relationships(rel.concept_to, RelationshipType(relationship_type))
                
                for indirect_rel in indirect_rels:
                    # Check if relationship already exists
                    existing = self.neurosql.find_relationships(start_concept, RelationshipType(relationship_type))
                    exists = any(r.concept_to == indirect_rel.concept_to for r in existing)
                    
                    if not exists:
                        inferred_rel = WeightedRelationship(
                            concept_from=start_concept,
                            concept_to=indirect_rel.concept_to,
                            relationship_type=RelationshipType(relationship_type),
                            weight=rel.weight * indirect_rel.weight * 0.8,
                            confidence=0.75,
                            metadata={'inferred': True, 'operator': 'transitive_closure'}
                        )
                        new_relationships.append(inferred_rel)
                        reasoning_steps.append(f"{start_concept} → {indirect_rel.concept_to}")
        
        return InferenceResult(
            relationships=new_relationships,
            confidence=0.75,
            reasoning_steps=reasoning_steps,
            operator=ReasoningOperator.TRANSITIVE_CLOSURE
        )
    
    def _property_inheritance(self, params: Dict[str, Any]):
        from neurosql_core import RelationshipType, WeightedRelationship
        
        concept_name = params.get('concept', '')
        
        if not concept_name or concept_name not in self.neurosql.concepts:
            return InferenceResult([], 0.0, [], ReasoningOperator.PROPERTY_INHERITANCE)
        
        new_relationships = []
        reasoning_steps = []
        
        # Get IS_A ancestors
        is_a_rels = self.neurosql.find_relationships(concept_name, RelationshipType.IS_A)
        
        for parent_rel in is_a_rels:
            parent = parent_rel.concept_to
            
            # Get properties of parent
            parent_props = self.neurosql.find_relationships(parent, RelationshipType.HAS_PROPERTY)
            
            for prop_rel in parent_props:
                property_name = prop_rel.concept_to
                
                # Check if concept already has this property
                existing_props = self.neurosql.find_relationships(concept_name, RelationshipType.HAS_PROPERTY)
                has_property = any(p.concept_to == property_name for p in existing_props)
                
                if not has_property:
                    inherited_rel = WeightedRelationship(
                        concept_from=concept_name,
                        concept_to=property_name,
                        relationship_type=RelationshipType.HAS_PROPERTY,
                        weight=prop_rel.weight * 0.85,
                        confidence=0.8,
                        metadata={'inferred': True, 'operator': 'property_inheritance', 'source': parent}
                    )
                    new_relationships.append(inherited_rel)
                    reasoning_steps.append(f"Inherited '{property_name}' from '{parent}'")
        
        return InferenceResult(
            relationships=new_relationships,
            confidence=0.8,
            reasoning_steps=reasoning_steps,
            operator=ReasoningOperator.PROPERTY_INHERITANCE
        )
    
    def _default_reasoning(self, params: Dict[str, Any]):
        from neurosql_core import RelationshipType, WeightedRelationship
        
        concept_name = params.get('concept', '')
        
        if not concept_name or concept_name not in self.neurosql.concepts:
            return InferenceResult([], 0.0, [], ReasoningOperator.DEFAULT_REASONING)
        
        concept = self.neurosql.get_concept(concept_name)
        domain = params.get('domain', concept.domain)
        
        defaults = {
            'cs': [('has_property', 'digital', 0.8), ('related_to', 'technology', 0.7)],
            'pets': [('has_property', 'domesticated', 0.85), ('related_to', 'animals', 0.9)],
        }
        
        new_relationships = []
        reasoning_steps = []
        
        if domain in defaults:
            for rel_type, target, weight in defaults[domain]:
                existing = self.neurosql.find_relationships(concept_name, RelationshipType(rel_type))
                exists = any(r.concept_to == target for r in existing)
                
                if not exists:
                    inferred_rel = WeightedRelationship(
                        concept_from=concept_name,
                        concept_to=target,
                        relationship_type=RelationshipType(rel_type),
                        weight=weight * 0.9,
                        confidence=0.7,
                        metadata={'inferred': True, 'operator': 'default_reasoning', 'domain': domain}
                    )
                    new_relationships.append(inferred_rel)
                    reasoning_steps.append(f"Default: {concept_name} → {target} ({rel_type})")
        
        return InferenceResult(
            relationships=new_relationships,
            confidence=0.7,
            reasoning_steps=reasoning_steps,
            operator=ReasoningOperator.DEFAULT_REASONING
        )
