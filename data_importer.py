import csv
import sqlite3
import json
from typing import List, Dict, Any
from neurosql_core import NeuroSQL, Concept, WeightedRelationship, RelationshipType


class DataImporter:
    """Import data from various sources into NeuroSQL"""
    
    def __init__(self, neurosql: NeuroSQL):
        self.neurosql = neurosql
    
    def import_from_csv(self, filepath: str, domain: str = "imported"):
        """
        Import concepts and relationships from CSV file.
        
        Expected CSV format:
        concept_name,attributes,abstraction_level,relationships
        
        Where attributes and relationships are JSON strings.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Create concept
                attributes = {}
                if row.get('attributes'):
                    try:
                        attributes = json.loads(row['attributes'])
                    except json.JSONDecodeError:
                        attributes = {"raw": row['attributes']}
                
                concept = Concept(
                    name=row['concept_name'],
                    attributes=attributes,
                    abstraction_level=int(row.get('abstraction_level', 0)),
                    domain=domain
                )
                self.neurosql.add_concept(concept)
                
                # Add relationships if specified
                if row.get('relationships'):
                    try:
                        rels = json.loads(row['relationships'])
                        for rel_data in rels:
                            relationship = WeightedRelationship(
                                concept_from=row['concept_name'],
                                concept_to=rel_data['to'],
                                relationship_type=RelationshipType(rel_data.get('type', 'related_to')),
                                weight=float(rel_data.get('weight', 1.0))
                            )
                            self.neurosql.add_weighted_relationship(relationship)
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Could not parse relationships for {row['concept_name']}: {e}")
    
    def import_from_sqlite(self, db_path: str, domain: str = "database"):
        """
        Import data from SQLite database.
        
        Expected tables:
        - concepts: name, attributes (JSON), abstraction_level
        - relationships: concept_from, concept_to, relationship_type, weight
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Import concepts
        try:
            cursor.execute("SELECT name, attributes, abstraction_level FROM concepts")
            for row in cursor.fetchall():
                attributes = {}
                if row[1]:
                    try:
                        attributes = json.loads(row[1])
                    except json.JSONDecodeError:
                        attributes = {}
                
                concept = Concept(
                    name=row[0],
                    attributes=attributes,
                    abstraction_level=row[2] if row[2] else 0,
                    domain=domain
                )
                self.neurosql.add_concept(concept)
        except sqlite3.OperationalError as e:
            print(f"Warning: Could not read concepts table: {e}")
        
        # Import relationships
        try:
            cursor.execute("SELECT concept_from, concept_to, relationship_type, weight FROM relationships")
            for row in cursor.fetchall():
                try:
                    rel_type = RelationshipType(row[2])
                except ValueError:
                    rel_type = RelationshipType.RELATED_TO
                
                relationship = WeightedRelationship(
                    concept_from=row[0],
                    concept_to=row[1],
                    relationship_type=rel_type,
                    weight=row[3] if row[3] else 1.0
                )
                self.neurosql.add_weighted_relationship(relationship)
        except sqlite3.OperationalError as e:
            print(f"Warning: Could not read relationships table: {e}")
        
        conn.close()
    
    def import_from_json(self, filepath: str, domain: str = "json"):
        """
        Import data from JSON file.
        
        Expected format:
        {
            "concepts": [
                {"name": "...", "attributes": {...}, "abstraction_level": 0}
            ],
            "relationships": [
                {"from": "...", "to": "...", "type": "is_a", "weight": 0.9}
            ]
        }
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Import concepts
        for item in data.get('concepts', []):
            concept = Concept(
                name=item['name'],
                attributes=item.get('attributes', {}),
                abstraction_level=item.get('abstraction_level', 0),
                domain=domain
            )
            self.neurosql.add_concept(concept)
        
        # Import relationships
        for item in data.get('relationships', []):
            try:
                rel_type = RelationshipType(item.get('type', 'related_to'))
            except ValueError:
                rel_type = RelationshipType.RELATED_TO
            
            relationship = WeightedRelationship(
                concept_from=item['from'],
                concept_to=item['to'],
                relationship_type=rel_type,
                weight=item.get('weight', 1.0)
            )
            self.neurosql.add_weighted_relationship(relationship)
    
    def import_from_json_api(self, api_endpoint: str, domain: str = "api"):
        """Import data from JSON API endpoint"""
        try:
            import requests
            
            response = requests.get(api_endpoint, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get('concepts', []):
                concept = Concept(
                    name=item['name'],
                    attributes=item.get('attributes', {}),
                    abstraction_level=item.get('abstraction_level', 0),
                    domain=domain
                )
                self.neurosql.add_concept(concept)
            
            for item in data.get('relationships', []):
                try:
                    rel_type = RelationshipType(item.get('type', 'related_to'))
                except ValueError:
                    rel_type = RelationshipType.RELATED_TO
                
                relationship = WeightedRelationship(
                    concept_from=item['from'],
                    concept_to=item['to'],
                    relationship_type=rel_type,
                    weight=item.get('weight', 1.0)
                )
                self.neurosql.add_weighted_relationship(relationship)
                
        except ImportError:
            print("Install requests: pip install requests")
        except Exception as e:
            print(f"Error importing from API: {e}")
    
    def export_to_csv(self, concepts_file: str, relationships_file: str):
        """Export data to CSV files"""
        # Export concepts
        with open(concepts_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['concept_name', 'attributes', 'abstraction_level', 'domain'])
            
            for name, concept in self.neurosql.concepts.items():
                writer.writerow([
                    concept.name,
                    json.dumps(concept.attributes),
                    concept.abstraction_level,
                    concept.domain
                ])
        
        # Export relationships
        with open(relationships_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['concept_from', 'concept_to', 'relationship_type', 'weight', 'confidence'])
            
            for rel in self.neurosql.relationships:
                writer.writerow([
                    rel.concept_from,
                    rel.concept_to,
                    rel.relationship_type.value,
                    rel.weight,
                    rel.confidence
                ])
        
        print(f"Exported to {concepts_file} and {relationships_file}")
    
    def export_to_sqlite(self, db_path: str):
        """Export data to SQLite database"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.executescript("""
            DROP TABLE IF EXISTS concepts;
            DROP TABLE IF EXISTS relationships;
            
            CREATE TABLE concepts (
                name TEXT PRIMARY KEY,
                attributes TEXT,
                abstraction_level INTEGER,
                domain TEXT
            );
            
            CREATE TABLE relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_from TEXT,
                concept_to TEXT,
                relationship_type TEXT,
                weight REAL,
                confidence REAL,
                FOREIGN KEY (concept_from) REFERENCES concepts(name),
                FOREIGN KEY (concept_to) REFERENCES concepts(name)
            );
        """)
        
        # Insert concepts
        for name, concept in self.neurosql.concepts.items():
            cursor.execute(
                "INSERT INTO concepts (name, attributes, abstraction_level, domain) VALUES (?, ?, ?, ?)",
                (concept.name, json.dumps(concept.attributes), concept.abstraction_level, concept.domain)
            )
        
        # Insert relationships
        for rel in self.neurosql.relationships:
            cursor.execute(
                "INSERT INTO relationships (concept_from, concept_to, relationship_type, weight, confidence) VALUES (?, ?, ?, ?, ?)",
                (rel.concept_from, rel.concept_to, rel.relationship_type.value, rel.weight, rel.confidence)
            )
        
        conn.commit()
        conn.close()
        print(f"Exported to {db_path}")