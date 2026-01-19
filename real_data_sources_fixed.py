# ============================================================================
# REAL DATA SOURCE CONNECTORS
# ============================================================================

import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
import json
import time
from datetime import datetime, timedelta
import hashlib
import sqlite3
from pathlib import Path
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# PUBMED API CLIENT
# ============================================================================

class PubMedClient:
    """Real PubMed API client with caching"""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, api_key: Optional[str] = None, email: str = "neuro@example.com"):
        self.api_key = api_key
        self.email = email
        self.cache_db = "data/pubmed_cache.db"
        self._init_cache()
        
    def _init_cache(self):
        """Initialize PubMed cache database"""
        Path("data").mkdir(exist_ok=True)
        
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pubmed_cache (
                    pmid TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    authors TEXT,
                    journal TEXT,
                    pub_date TEXT,
                    doi TEXT,
                    retrieved_at TIMESTAMP,
                    query_hash TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    query_hash TEXT PRIMARY KEY,
                    pmids TEXT,
                    total_count INTEGER,
                    retrieved_at TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search PubMed for neuroscience literature"""
        start_time = time.time()
        
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Check cache first
        cached = self._get_cached_search(query_hash)
        if cached:
            logger.info(f"Cache hit for query: {query}")
            return cached
        
        logger.info(f"Querying PubMed: {query}")
        
        try:
            # Build PubMed query
            params = {
                'db': 'pubmed',
                'term': f"{query}[Title/Abstract]",
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'relevance'
            }
            
            if self.api_key:
                params['api_key'] = self.api_key
            if self.email:
                params['email'] = self.email
            
            # Search PubMed
            search_url = f"{self.BASE_URL}/esearch.fcgi"
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            search_data = response.json()
            pmids = search_data.get('esearchresult', {}).get('idlist', [])
            
            if not pmids:
                return []
            
            # Fetch details for each PMID
            articles = self._fetch_article_details(pmids[:max_results])
            
            # Cache results
            self._cache_search(query_hash, pmids, articles)
            
            query_time = time.time() - start_time
            logger.info(f"PubMed search completed in {query_time:.2f}s, found {len(articles)} articles")
            
            return articles
            
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []
    
    def _fetch_article_details(self, pmids: List[str]) -> List[Dict]:
        """Fetch detailed article information"""
        if not pmids:
            return []
        
        # Check cache for individual articles
        articles = []
        pmids_to_fetch = []
        
        for pmid in pmids:
            cached = self._get_cached_article(pmid)
            if cached:
                articles.append(cached)
            else:
                pmids_to_fetch.append(pmid)
        
        if not pmids_to_fetch:
            return articles
        
        try:
            # Fetch multiple articles at once
            pmid_str = ",".join(pmids_to_fetch)
            fetch_url = f"{self.BASE_URL}/esummary.fcgi"
            params = {
                'db': 'pubmed',
                'id': pmid_str,
                'retmode': 'json'
            }
            
            response = requests.get(fetch_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('result', {})
            
            for pmid in pmids_to_fetch:
                if pmid in results:
                    article_data = results[pmid]
                    
                    article = {
                        'pmid': pmid,
                        'title': article_data.get('title', ''),
                        'abstract': article_data.get('abstract', ''),
                        'authors': article_data.get('authors', []),
                        'journal': article_data.get('source', ''),
                        'pub_date': article_data.get('pubdate', ''),
                        'doi': article_data.get('doi', ''),
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    }
                    
                    articles.append(article)
                    
                    # Cache this article
                    self._cache_article(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Article fetch error: {e}")
            return articles
    
    def _get_cached_search(self, query_hash: str) -> Optional[List[Dict]]:
        """Get cached search results"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute(
                    "SELECT pmids FROM search_cache WHERE query_hash = ?",
                    (query_hash,)
                )
                row = cursor.fetchone()
                
                if row:
                    pmids = json.loads(row[0])
                    articles = []
                    
                    for pmid in pmids:
                        article = self._get_cached_article(pmid)
                        if article:
                            articles.append(article)
                    
                    return articles if articles else None
                    
        except Exception as e:
            logger.error(f"Cache read error: {e}")
        
        return None
    
    def _cache_search(self, query_hash: str, pmids: List[str], articles: List[Dict]):
        """Cache search results"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO search_cache (query_hash, pmids, total_count, retrieved_at) VALUES (?, ?, ?, ?)",
                    (query_hash, json.dumps(pmids), len(articles), datetime.now().isoformat())
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    def _get_cached_article(self, pmid: str) -> Optional[Dict]:
        """Get cached article"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute(
                    "SELECT title, abstract, authors, journal, pub_date, doi FROM pubmed_cache WHERE pmid = ?",
                    (pmid,)
                )
                row = cursor.fetchone()
                
                if row:
                    return {
                        'pmid': pmid,
                        'title': row[0],
                        'abstract': row[1],
                        'authors': json.loads(row[2]) if row[2] else [],
                        'journal': row[3],
                        'pub_date': row[4],
                        'doi': row[5],
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        'cached': True
                    }
                    
        except Exception as e:
            logger.error(f"Article cache read error: {e}")
        
        return None
    
    def _cache_article(self, article: Dict):
        """Cache an article"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO pubmed_cache 
                    (pmid, title, abstract, authors, journal, pub_date, doi, retrieved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article['pmid'],
                    article.get('title', ''),
                    article.get('abstract', ''),
                    json.dumps(article.get('authors', [])),
                    article.get('journal', ''),
                    article.get('pub_date', ''),
                    article.get('doi', ''),
                    datetime.now().isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Article cache write error: {e}")
    
    def extract_relationships(self, pmid: str) -> List[Dict]:
        """Extract relationships from PubMed article (simplified)"""
        article = self._get_cached_article(pmid)
        if not article:
            article = self._fetch_article_details([pmid])[0]
        
        if not article:
            return []
        
        # Simple relationship extraction from title/abstract
        text = (article.get('title', '') + " " + article.get('abstract', '')).lower()
        
        relationships = []
        
        # Look for common patterns
        patterns = [
            (r'(\w+)\s+(modulates|regulates|affects|influences)\s+(\w+)', 'MODULATES'),
            (r'(\w+)\s+(inhibits|suppresses|blocks)\s+(\w+)', 'INHIBITS'),
            (r'(\w+)\s+(activates|excites|stimulates)\s+(\w+)', 'ACTIVATES'),
            (r'(\w+)\s+(is associated with|correlates with)\s+(\w+)', 'ASSOCIATED_WITH'),
        ]
        
        for pattern, rel_type in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                subject = match.group(1)
                object = match.group(3)
                
                # Filter out common stop words
                stop_words = {'the', 'and', 'of', 'in', 'to', 'with', 'for', 'on', 'by'}
                if subject not in stop_words and object not in stop_words:
                    relationships.append({
                        'subject': subject,
                        'predicate': rel_type,
                        'object': object,
                        'source': f"PMID:{pmid}",
                        'snippet': text[max(0, match.start()-50):match.end()+50],
                        'confidence': 0.7
                    })
        
        return relationships

# ============================================================================
# NEUROSYNTH API CLIENT
# ============================================================================

class NeurosynthClient:
    """Neurosynth API client for fMRI meta-analysis data"""
    
    BASE_URL = "https://neurosynth.org/api/v2"
    
    def __init__(self):
        self.cache = {}
        logger.info("Neurosynth client initialized")
    
    def get_term_associations(self, term: str, limit: int = 10) -> List[Dict]:
        """Get brain regions associated with a term"""
        try:
            url = f"{self.BASE_URL}/analyses/terms/{term}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            associations = []
            if 'data' in data and 'values' in data['data']:
                regions = data['data']['regions']
                values = data['data']['values']
                
                for region, value in zip(regions[:limit], values[:limit]):
                    associations.append({
                        'term': term,
                        'brain_region': region,
                        'association_strength': float(value),
                        'source': 'neurosynth',
                        'url': f"https://neurosynth.org/analyses/{term}"
                    })
            
            return associations
            
        except Exception as e:
            logger.error(f"Neurosynth API error: {e}")
            return []
    
    def get_region_activations(self, region: str) -> List[Dict]:
        """Get terms associated with a brain region"""
        try:
            url = f"{self.BASE_URL}/regions/{region}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            activations = []
            if 'data' in data:
                for term, stats in data['data'].items():
                    if isinstance(stats, dict) and 'pFgA' in stats:
                        activations.append({
                            'brain_region': region,
                            'term': term,
                            'probability': float(stats['pFgA']),
                            'studies': int(stats.get('count', 0)),
                            'source': 'neurosynth'
                        })
            
            return sorted(activations, key=lambda x: x['probability'], reverse=True)[:10]
            
        except Exception as e:
            logger.error(f"Neurosynth region API error: {e}")
            return []

# ============================================================================
# ALLEN BRAIN ATLAS CLIENT
# ============================================================================

class AllenBrainClient:
    """Allen Brain Atlas API client"""
    
    BASE_URL = "https://api.brain-map.org/api/v2/data"
    
    def __init__(self):
        self.structure_cache = {}
        logger.info("Allen Brain client initialized")
    
    def get_brain_structure(self, structure_name: str) -> Optional[Dict]:
        """Get information about a brain structure"""
        try:
            # Search for structure
            search_url = f"{self.BASE_URL}/Structure/query.json"
            params = {
                'criteria': f"[name$il'{structure_name}']",
                'include': 'graphic_group_label'
            }
            
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data['success'] and data['msg']:
                structure = data['msg'][0]
                
                return {
                    'id': structure['id'],
                    'name': structure['name'],
                    'acronym': structure['acronym'],
                    'ontology': structure.get('graphic_group_label', ''),
                    'parent_structure': structure.get('parent_structure_id', ''),
                    'description': structure.get('description', ''),
                    'source': 'allen_brain_atlas',
                    'url': f"https://mouse.brain-map.org/structure/{structure['id']}"
                }
            
        except Exception as e:
            logger.error(f"Allen Brain API error: {e}")
        
        return None
    
    def get_structure_connectivity(self, structure_id: int) -> List[Dict]:
        """Get connectivity data for a brain structure"""
        try:
            url = f"{self.BASE_URL}/Connection/query.json"
            params = {
                'criteria': f"[structure_id$eq{structure_id}]",
                'include': 'structure,connection_strength'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            connections = []
            if data['success'] and data['msg']:
                for conn in data['msg']:
                    connections.append({
                        'source_structure': structure_id,
                        'target_structure': conn.get('target_structure_id'),
                        'strength': conn.get('connection_strength', 0),
                        'method': conn.get('method', ''),
                        'source': 'allen_brain_connectivity'
                    })
            
            return connections
            
        except Exception as e:
            logger.error(f"Allen connectivity API error: {e}")
            return []

# ============================================================================
# REAL EVIDENCE DATABASE
# ============================================================================

class EvidenceDatabase:
    """Real evidence database with versioning"""
    
    def __init__(self, db_path: str = "data/evidence.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize evidence database schema"""
        Path("data").mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Evidence table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    snippet TEXT NOT NULL,
                    extracted_fact TEXT,
                    retrieval_time TIMESTAMP NOT NULL,
                    content_hash TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    validation_score REAL DEFAULT 0.0,
                    version INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Relationships table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    relationship_id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    evidence_ids TEXT,  -- JSON array of evidence IDs
                    confidence REAL DEFAULT 0.5,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Usage tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence_usage (
                    usage_id TEXT PRIMARY KEY,
                    evidence_id TEXT NOT NULL,
                    inference_id TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    contribution_weight REAL DEFAULT 1.0,
                    used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id)
                )
            """)
            
            # Version history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS version_history (
                    version_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    previous_version INTEGER,
                    new_version INTEGER,
                    change_type TEXT,
                    change_description TEXT,
                    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    changed_by TEXT DEFAULT 'system'
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_source ON evidence(source_type, source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_subject ON relationships(subject)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_object ON relationships(object)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_inference ON evidence_usage(inference_id)")
            
            conn.commit()
    
    def store_evidence(self, evidence: Dict) -> str:
        """Store evidence with versioning"""
        evidence_id = evidence.get('evidence_id', f"ev_{hashlib.md5(str(evidence).encode()).hexdigest()[:16]}")
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if evidence already exists
            cursor = conn.execute(
                "SELECT evidence_id, version FROM evidence WHERE content_hash = ?",
                (evidence.get('content_hash', ''),)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing evidence
                new_version = existing[1] + 1
                conn.execute("""
                    UPDATE evidence 
                    SET validation_score = ?, updated_at = CURRENT_TIMESTAMP, version = ?
                    WHERE evidence_id = ?
                """, (evidence.get('validation_score', 0.0), new_version, existing[0]))
                
                # Log version change
                self._log_version_change('evidence', existing[0], existing[1], new_version, 'update')
                
                return existing[0]
            else:
                # Insert new evidence
                conn.execute("""
                    INSERT INTO evidence 
                    (evidence_id, source_type, source_id, snippet, extracted_fact, 
                     retrieval_time, content_hash, confidence, validation_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    evidence_id,
                    evidence.get('source_type', 'unknown'),
                    evidence.get('source_id', ''),
                    evidence.get('snippet', ''),
                    json.dumps(evidence.get('extracted_fact', {})),
                    evidence.get('retrieval_time', datetime.now().isoformat()),
                    evidence.get('content_hash', hashlib.sha256(evidence.get('snippet', '').encode()).hexdigest()),
                    evidence.get('confidence', 0.5),
                    evidence.get('validation_score', 0.0)
                ))
                
                # Log version change
                self._log_version_change('evidence', evidence_id, 0, 1, 'create')
                
                return evidence_id
    
    def _log_version_change(self, entity_type: str, entity_id: str, 
                          prev_version: int, new_version: int, 
                          change_type: str, description: str = ""):
        """Log version changes"""
        version_id = "ver_{hashlib.md5(f"{entity_type}_{entity_id}_{new_version}".encode()).hexdigest()[:16]}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO version_history 
                (version_id, entity_type, entity_id, previous_version, new_version, change_type, change_description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (version_id, entity_type, entity_id, prev_version, new_version, change_type, description))
    
    def get_evidence_for_relationship(self, subject: str, predicate: str, object: str) -> List[Dict]:
        """Get evidence supporting a specific relationship"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM relationships 
                WHERE subject = ? AND predicate = ? AND object = ?
                ORDER BY confidence DESC
            """, (subject, predicate, object))
            
            relationships = [dict(row) for row in cursor.fetchall()]
            
            evidence_list = []
            for rel in relationships:
                if rel['evidence_ids']:
                    evidence_ids = json.loads(rel['evidence_ids'])
                    for ev_id in evidence_ids:
                        evidence = self.get_evidence_by_id(ev_id)
                        if evidence:
                            evidence_list.append(evidence)
            
            return evidence_list
    
    def get_evidence_by_id(self, evidence_id: str) -> Optional[Dict]:
        """Get evidence by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM evidence WHERE evidence_id = ?",
                (evidence_id,)
            )
            row = cursor.fetchone()
            
            if row:
                evidence = dict(row)
                if evidence.get('extracted_fact'):
                    evidence['extracted_fact'] = json.loads(evidence['extracted_fact'])
                return evidence
        
        return None
    
    def record_evidence_usage(self, evidence_id: str, inference_id: str, 
                            query_hash: str, weight: float = 1.0):
        """Record when evidence is used in an inference"""
        usage_id = "use_{hashlib.md5(f"{evidence_id}_{inference_id}".encode()).hexdigest()[:16]}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO evidence_usage 
                (usage_id, evidence_id, inference_id, query_hash, contribution_weight)
                VALUES (?, ?, ?, ?, ?)
            """, (usage_id, evidence_id, inference_id, query_hash, weight))
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM evidence")
            evidence_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM relationships")
            relationship_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM evidence_usage")
            usage_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM version_history")
            version_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT AVG(confidence) FROM evidence")
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            cursor = conn.execute("SELECT AVG(validation_score) FROM evidence")
            avg_validation = cursor.fetchone()[0] or 0.0
        
        return {
            'evidence_count': evidence_count,
            'relationship_count': relationship_count,
            'usage_count': usage_count,
            'version_count': version_count,
            'average_confidence': avg_confidence,
            'average_validation': avg_validation,
            'database_size_mb': Path(self.db_path).stat().st_size / (1024 * 1024) if Path(self.db_path).exists() else 0
        }
