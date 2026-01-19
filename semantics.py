# semantics.py
from typing import Dict, List, Any, Set, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

class SemanticProperty(Enum):
    TRANSITIVE = "transitive"
    SYMMETRIC = "symmetric"
    REFLEXIVE = "reflexive"
    ANTI_SYMMETRIC = "anti_symmetric"
    INHERITABLE = "inheritable"

@dataclass
class RelationshipSchema:
    name: str
    domain: str
    range: str
    properties: Set[SemanticProperty] = field(default_factory=set)
    inverse: Optional[str] = None
    description: str = ""
    
    def has_property(self, property: SemanticProperty) -> bool:
        return property in self.properties

class SemanticsEngine:
    def __init__(self):
        self.schemas: Dict[str, RelationshipSchema] = {}
        self._setup_default_schemas()
    
    def _setup_default_schemas(self):
        self.schemas["is_a"] = RelationshipSchema(
            name="is_a",
            domain="*",
            range="*",
            properties={SemanticProperty.TRANSITIVE, SemanticProperty.INHERITABLE},
            inverse="has_instance",
            description="Subclass/instance relationship"
        )
        
        self.schemas["has_property"] = RelationshipSchema(
            name="has_property",
            domain="*",
            range="property",
            properties={SemanticProperty.INHERITABLE},
            description="Object has a property"
        )
        
        self.schemas["related_to"] = RelationshipSchema(
            name="related_to",
            domain="*",
            range="*",
            properties={SemanticProperty.SYMMETRIC},
            inverse="related_to",
            description="General symmetric association"
        )
    
    def validate_relationship(self, relationship) -> List[str]:
        errors = []
        schema = self.schemas.get(relationship.relationship_type.value)
        
        if not schema:
            errors.append(f"No schema for relationship type: {relationship.relationship_type.value}")
        
        return errors
