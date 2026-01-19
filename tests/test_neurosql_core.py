"""
Unit tests for NeuroSQL Core module.

Run with: python -m pytest tests/test_neurosql_core.py -v
Or: python tests/test_neurosql_core.py
"""
import unittest
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurosql_core import NeuroSQL, Concept, WeightedRelationship, RelationshipType


class TestConcept(unittest.TestCase):
    """Test Concept dataclass"""
    
    def test_create_concept_minimal(self):
        concept = Concept("Test")
        self.assertEqual(concept.name, "Test")
        self.assertEqual(concept.attributes, {})
        self.assertEqual(concept.abstraction_level, 0)
        self.assertEqual(concept.domain, "general")
    
    def test_create_concept_full(self):
        concept = Concept(
            name="TestConcept",
            attributes={"key": "value", "number": 42},
            abstraction_level=2,
            domain="test_domain"
        )
        self.assertEqual(concept.name, "TestConcept")
        self.assertEqual(concept.attributes["key"], "value")
        self.assertEqual(concept.abstraction_level, 2)
        self.assertEqual(concept.domain, "test_domain")


class TestWeightedRelationship(unittest.TestCase):
    """Test WeightedRelationship dataclass"""
    
    def test_create_relationship_minimal(self):
        rel = WeightedRelationship("A", "B", RelationshipType.IS_A)
        self.assertEqual(rel.concept_from, "A")
        self.assertEqual(rel.concept_to, "B")
        self.assertEqual(rel.relationship_type, RelationshipType.IS_A)
        self.assertEqual(rel.weight, 1.0)
        self.assertEqual(rel.confidence, 1.0)
    
    def test_create_relationship_full(self):
        rel = WeightedRelationship(
            concept_from="A",
            concept_to="B",
            relationship_type=RelationshipType.CAUSES,
            weight=0.8,
            confidence=0.9,
            metadata={"source": "test"}
        )
        self.assertEqual(rel.weight, 0.8)
        self.assertEqual(rel.confidence, 0.9)
        self.assertEqual(rel.metadata["source"], "test")


class TestNeuroSQL(unittest.TestCase):
    """Test NeuroSQL class"""
    
    def setUp(self):
        self.neurosql = NeuroSQL("TestGraph")
    
    def test_init(self):
        self.assertEqual(self.neurosql.name, "TestGraph")
        self.assertEqual(len(self.neurosql.concepts), 0)
        self.assertEqual(len(self.neurosql.relationships), 0)
    
    def test_add_concept(self):
        concept = Concept("Test", {"key": "value"}, 1, "test")
        self.neurosql.add_concept(concept)
        
        self.assertIn("Test", self.neurosql.concepts)
        self.assertEqual(self.neurosql.concepts["Test"].name, "Test")
    
    def test_add_concept_with_layer(self):
        concept = Concept("Concrete", {}, 0, "test")
        self.neurosql.add_concept(concept, layer="concrete")
        
        self.assertIn("Concrete", self.neurosql.layers["concrete"])
    
    def test_get_concept(self):
        concept = Concept("Test", {"key": "value"})
        self.neurosql.add_concept(concept)
        
        retrieved = self.neurosql.get_concept("Test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Test")
        
        # Test non-existent concept
        self.assertIsNone(self.neurosql.get_concept("NonExistent"))
    
    def test_add_weighted_relationship(self):
        concept1 = Concept("A")
        concept2 = Concept("B")
        self.neurosql.add_concept(concept1)
        self.neurosql.add_concept(concept2)
        
        rel = WeightedRelationship("A", "B", RelationshipType.IS_A, 0.9)
        self.neurosql.add_weighted_relationship(rel)
        
        self.assertEqual(len(self.neurosql.relationships), 1)
        self.assertEqual(self.neurosql.relationships[0].weight, 0.9)
    
    def test_find_relationships(self):
        # Setup
        for name in ["A", "B", "C"]:
            self.neurosql.add_concept(Concept(name))
        
        rel1 = WeightedRelationship("A", "B", RelationshipType.IS_A, 0.9)
        rel2 = WeightedRelationship("A", "C", RelationshipType.RELATED_TO, 0.5)
        self.neurosql.add_weighted_relationship(rel1)
        self.neurosql.add_weighted_relationship(rel2)
        
        # Test find all
        rels = self.neurosql.find_relationships("A")
        self.assertEqual(len(rels), 2)
        
        # Test filter by type
        is_a_rels = self.neurosql.find_relationships("A", RelationshipType.IS_A)
        self.assertEqual(len(is_a_rels), 1)
        self.assertEqual(is_a_rels[0].relationship_type, RelationshipType.IS_A)
        
        # Test filter by weight
        strong_rels = self.neurosql.find_relationships("A", min_weight=0.8)
        self.assertEqual(len(strong_rels), 1)
        self.assertGreaterEqual(strong_rels[0].weight, 0.8)
    
    def test_get_abstraction_hierarchy(self):
        # Create hierarchy
        concrete = Concept("Concrete", {}, 0)
        abstract = Concept("Abstract", {}, 1)
        meta = Concept("Meta", {}, 2)
        
        for concept in [concrete, abstract, meta]:
            self.neurosql.add_concept(concept)
        
        self.neurosql.add_weighted_relationship(
            WeightedRelationship("Concrete", "Abstract", RelationshipType.IS_A)
        )
        self.neurosql.add_weighted_relationship(
            WeightedRelationship("Concrete", "Meta", RelationshipType.IS_A)
        )
        
        hierarchy = self.neurosql.get_abstraction_hierarchy("Concrete")
        self.assertEqual(len(hierarchy), 2)
        # Should be sorted by abstraction level
        self.assertEqual(hierarchy[0].name, "Abstract")
        self.assertEqual(hierarchy[1].name, "Meta")
    
    def test_save_and_load(self):
        # Setup data
        concept = Concept("Test", {"key": "value"}, 1, "test")
        rel = WeightedRelationship("A", "B", RelationshipType.IS_A, 0.8)
        
        self.neurosql.add_concept(concept)
        self.neurosql.add_weighted_relationship(rel)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name
        
        try:
            self.neurosql.save_to_file(temp_file)
            
            # Load into new instance
            new_neurosql = NeuroSQL("LoadedGraph")
            new_neurosql.load_from_file(temp_file)
            
            # Verify loaded data
            loaded_concept = new_neurosql.get_concept("Test")
            self.assertIsNotNone(loaded_concept)
            self.assertEqual(loaded_concept.attributes["key"], "value")
            self.assertEqual(loaded_concept.abstraction_level, 1)
            
            self.assertEqual(len(new_neurosql.relationships), 1)
            self.assertEqual(new_neurosql.relationships[0].weight, 0.8)
            
        finally:
            os.unlink(temp_file)


class TestRelationshipType(unittest.TestCase):
    """Test RelationshipType enum"""
    
    def test_relationship_types(self):
        self.assertEqual(RelationshipType.IS_A.value, "is_a")
        self.assertEqual(RelationshipType.PART_OF.value, "part_of")
        self.assertEqual(RelationshipType.HAS_PROPERTY.value, "has_property")
        self.assertEqual(RelationshipType.CAUSES.value, "causes")
        self.assertEqual(RelationshipType.USES.value, "uses")
        self.assertEqual(RelationshipType.RELATED_TO.value, "related_to")
    
    def test_relationship_type_from_string(self):
        rel_type = RelationshipType("is_a")
        self.assertEqual(rel_type, RelationshipType.IS_A)


if __name__ == '__main__':
    unittest.main(verbosity=2)