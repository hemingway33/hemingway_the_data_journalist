"""
Neo4j-backed RDF Triple Store Implementation

This module implements an RDF triple store using Neo4j as the backend database.
It supports storing, querying, and managing RDF triples (subject, predicate, object).
"""

from neo4j import GraphDatabase
from typing import List, Tuple, Optional, Dict, Any


class Neo4jTripleStore:
    """A Neo4j-backed RDF triple store implementation."""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", password: str = None, 
                 database: str = "neo4j"):
        """
        Initialize the triple store with Neo4j connection details.
        
        Args:
            uri: Neo4j connection URI (e.g., bolt://localhost:7687)
            user: Neo4j username
            password: Neo4j password
            database: Neo4j database name
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
    
    def connect(self) -> None:
        """Establish connection to the Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")
    
    def disconnect(self) -> None:
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
    
    def initialize_store(self) -> None:
        """Create constraints and indexes needed for the triple store."""
        if not self.driver:
            self.connect()
            
        try:
            with self.driver.session(database=self.database) as session:
                # Create constraint on URI nodes to ensure uniqueness
                session.run("""
                    CREATE CONSTRAINT uri_uniqueness IF NOT EXISTS
                    FOR (u:URI) REQUIRE u.value IS UNIQUE
                """)
                
                # Create indexes for efficient querying
                session.run("""
                    CREATE INDEX triple_subject IF NOT EXISTS
                    FOR ()-[r:SUBJECT]->()
                    ON (r.graph)
                """)
                
                session.run("""
                    CREATE INDEX triple_predicate IF NOT EXISTS
                    FOR ()-[r:PREDICATE]->()
                    ON (r.graph)
                """)
                
                session.run("""
                    CREATE INDEX triple_object IF NOT EXISTS
                    FOR ()-[r:OBJECT]->()
                    ON (r.graph, r.is_literal)
                """)
        except Exception as e:
            raise Exception(f"Failed to initialize triple store schema: {e}")
    
    def add_triple(self, subject: str, predicate: str, object_: str, 
                  is_literal: bool = False, graph: Optional[str] = None) -> None:
        """
        Add a triple to the store.
        
        Args:
            subject: Subject URI
            predicate: Predicate URI
            object_: Object URI or literal value
            is_literal: Whether the object is a literal value
            graph: Optional named graph URI
        """
        if not self.driver:
            self.connect()
            
        try:
            with self.driver.session(database=self.database) as session:
                # In Neo4j, we'll represent a triple as a node with relationships to the URIs
                # This pattern allows for efficient querying of triple patterns
                
                # The default graph if none specified
                graph_value = graph if graph else "default"
                
                # Create a Cypher query that:
                # 1. Creates or finds the URI nodes for subject, predicate, object
                # 2. Creates a Triple node that links to these URI nodes
                query = """
                MERGE (s:URI {value: $subject})
                MERGE (p:URI {value: $predicate})
                
                // Handle literal vs URI objects differently
                CALL {
                    WITH s, p
                    WITH s, p
                    MERGE (o:URI {value: $object})
                    WITH s, p, o
                    CREATE (t:Triple {id: randomUUID()})
                    CREATE (t)-[:SUBJECT {graph: $graph}]->(s)
                    CREATE (t)-[:PREDICATE {graph: $graph}]->(p)
                    CREATE (t)-[:OBJECT {graph: $graph, is_literal: $is_literal}]->(o)
                }
                """
                
                session.run(
                    query,
                    subject=subject,
                    predicate=predicate, 
                    object=object_,
                    is_literal=is_literal,
                    graph=graph_value
                )
                
        except Exception as e:
            raise Exception(f"Failed to add triple: {e}")
    
    def add_triples(self, triples: List[Tuple[str, str, str, bool, Optional[str]]]) -> None:
        """
        Add multiple triples to the store in a single transaction.
        
        Args:
            triples: List of tuples (subject, predicate, object, is_literal, graph)
        """
        if not self.driver:
            self.connect()
            
        try:
            with self.driver.session(database=self.database) as session:
                for subject, predicate, object_, is_literal, graph in triples:
                    self.add_triple(subject, predicate, object_, is_literal, graph)
                    
        except Exception as e:
            raise Exception(f"Failed to add triples batch: {e}")
    
    def remove_triple(self, subject: Optional[str] = None, predicate: Optional[str] = None, 
                     object_: Optional[str] = None, graph: Optional[str] = None) -> int:
        """
        Remove triples matching the specified pattern.
        
        Args:
            subject: Optional subject URI to match
            predicate: Optional predicate URI to match
            object_: Optional object URI/literal to match
            graph: Optional graph URI to match
            
        Returns:
            Number of triples removed
        """
        if not self.driver:
            self.connect()
            
        # At least one criteria must be specified
        if not any([subject, predicate, object_, graph]):
            raise ValueError("At least one triple component must be specified")
        
        # Default graph if not specified but other components are
        graph_value = graph if graph else "default"
        
        # Build the match part of the Cypher query
        match_parts = ["MATCH (t:Triple)"]
        where_parts = []
        params = {"graph": graph_value}
        
        if subject:
            match_parts.append("MATCH (t)-[:SUBJECT {graph: $graph}]->(s:URI)")
            where_parts.append("s.value = $subject")
            params["subject"] = subject
            
        if predicate:
            match_parts.append("MATCH (t)-[:PREDICATE {graph: $graph}]->(p:URI)")
            where_parts.append("p.value = $predicate")
            params["predicate"] = predicate
            
        if object_:
            match_parts.append("MATCH (t)-[:OBJECT {graph: $graph}]->(o:URI)")
            where_parts.append("o.value = $object")
            params["object"] = object_
        
        match_clause = " ".join(match_parts)
        where_clause = " AND ".join(where_parts)
        
        if where_clause:
            where_clause = "WHERE " + where_clause
        
        # Full query
        query = f"""
        {match_clause}
        {where_clause}
        WITH t
        DETACH DELETE t
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params)
                return result.consume().counters.nodes_deleted
                
        except Exception as e:
            raise Exception(f"Failed to remove triples: {e}")
    
    def query_triples(self, subject: Optional[str] = None, predicate: Optional[str] = None, 
                     object_: Optional[str] = None, graph: Optional[str] = None) -> List[Tuple[str, str, str, bool, Optional[str]]]:
        """
        Query triples matching the specified pattern.
        
        Args:
            subject: Optional subject URI to match
            predicate: Optional predicate URI to match
            object_: Optional object URI/literal to match
            graph: Optional graph URI to match
            
        Returns:
            List of matching triples (subject, predicate, object, is_literal, graph)
        """
        if not self.driver:
            self.connect()
            
        # Default graph if not specified
        graph_value = graph if graph else "default"
        
        # Build the match part of the Cypher query
        match_parts = ["MATCH (t:Triple)"]
        match_parts.append("MATCH (t)-[:SUBJECT {graph: $graph}]->(s:URI)")
        match_parts.append("MATCH (t)-[:PREDICATE {graph: $graph}]->(p:URI)")
        match_parts.append("MATCH (t)-[:OBJECT {graph: $graph}]->(o:URI)")
        
        where_parts = []
        params = {"graph": graph_value}
        
        if subject:
            where_parts.append("s.value = $subject")
            params["subject"] = subject
            
        if predicate:
            where_parts.append("p.value = $predicate")
            params["predicate"] = predicate
            
        if object_:
            where_parts.append("o.value = $object")
            params["object"] = object_
        
        match_clause = " ".join(match_parts)
        where_clause = " AND ".join(where_parts)
        
        if where_clause:
            where_clause = "WHERE " + where_clause
        
        # Full query
        query = f"""
        {match_clause}
        {where_clause}
        RETURN s.value AS subject, p.value AS predicate, o.value AS object,
               EXISTS((t)-[:OBJECT {{is_literal: true}}]->(o)) AS is_literal,
               $graph AS graph
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params)
                return [(record["subject"], record["predicate"], record["object"], 
                         record["is_literal"], record["graph"]) 
                        for record in result]
                
        except Exception as e:
            raise Exception(f"Failed to query triples: {e}")
    
    def cypher_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query against the graph.
        
        This provides full access to Neo4j's query language for advanced graph operations.
        
        Args:
            query: A Cypher query string
            params: Parameters for the Cypher query
            
        Returns:
            List of result records as dictionaries
        """
        if not self.driver:
            self.connect()
            
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params or {})
                return [dict(record) for record in result]
                
        except Exception as e:
            raise Exception(f"Failed to execute Cypher query: {e}")
    
    def count_triples(self, graph: Optional[str] = None) -> int:
        """
        Count the total number of triples in the store.
        
        Args:
            graph: Optional named graph to count triples in
            
        Returns:
            Count of triples
        """
        if not self.driver:
            self.connect()
            
        try:
            with self.driver.session(database=self.database) as session:
                query = "MATCH (t:Triple) RETURN count(t) AS count"
                params = {}
                
                if graph:
                    query = """
                    MATCH (t:Triple)-[:SUBJECT {graph: $graph}]->()
                    RETURN count(t) AS count
                    """
                    params["graph"] = graph
                    
                result = session.run(query, params)
                return result.single()["count"]
                
        except Exception as e:
            raise Exception(f"Failed to count triples: {e}")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        
    # Additional Neo4j-specific graph operations
    
    def get_connected_nodes(self, uri: str, relationship_type: Optional[str] = None,
                           max_depth: int = 1) -> List[Dict[str, Any]]:
        """
        Get nodes connected to the specified URI node up to a certain depth.
        
        Args:
            uri: URI value to start from
            relationship_type: Optional specific relationship type to traverse
            max_depth: Maximum traversal depth
            
        Returns:
            List of connected nodes with their relationships
        """
        if not self.driver:
            self.connect()
        
        rel_type = f":{relationship_type}" if relationship_type else ""
        
        query = f"""
        MATCH path = (start:URI {{value: $uri}})-[{rel_type}*1..{max_depth}]-(connected)
        RETURN path
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, uri=uri)
                return [dict(record) for record in result]
                
        except Exception as e:
            raise Exception(f"Failed to get connected nodes: {e}")
    
    def shortest_path(self, source_uri: str, target_uri: str, 
                     max_depth: int = 5) -> Optional[List[Dict[str, Any]]]:
        """
        Find the shortest path between two URI nodes.
        
        Args:
            source_uri: Starting URI value
            target_uri: Target URI value
            max_depth: Maximum path length to search
            
        Returns:
            Path if found, None otherwise
        """
        if not self.driver:
            self.connect()
        
        query = """
        MATCH path = shortestPath((source:URI {value: $source})-[*1..$max_depth]-(target:URI {value: $target}))
        RETURN path
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query, 
                    source=source_uri,
                    target=target_uri,
                    max_depth=max_depth
                )
                record = result.single()
                return dict(record) if record else None
                
        except Exception as e:
            raise Exception(f"Failed to find shortest path: {e}")


# Example usage
if __name__ == "__main__":
    # Create a triple store
    store = Neo4jTripleStore(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",  # Replace with actual password
        database="neo4j"
    )
    
    # Initialize the store
    store.initialize_store()
    
    # Add some example triples
    store.add_triple(
        subject="http://example.org/person/john",
        predicate="http://example.org/ontology/name",
        object_="John Smith",
        is_literal=True
    )
    
    store.add_triple(
        subject="http://example.org/person/john",
        predicate="http://example.org/ontology/age",
        object_="42",
        is_literal=True
    )
    
    store.add_triple(
        subject="http://example.org/person/john",
        predicate="http://example.org/ontology/friend",
        object_="http://example.org/person/jane",
        is_literal=False
    )
    
    # Query all triples about John
    results = store.query_triples(subject="http://example.org/person/john")
    
    print(f"Found {len(results)} triples about John:")
    for triple in results:
        subject, predicate, object_, is_literal, graph = triple
        print(f"{subject} {predicate} {object_}")
    
    # Count triples
    count = store.count_triples()
    print(f"Total triples in store: {count}")
    
    # Use Neo4j's native graph capabilities for path analysis
    path = store.shortest_path(
        source_uri="http://example.org/person/john",
        target_uri="http://example.org/person/jane"
    )
    
    if path:
        print("Found path between John and Jane")
    else:
        print("No path found between John and Jane")
    
    # Close connection
    store.disconnect()
