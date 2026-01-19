class Concept:
    """Represents a concept in the knowledge graph."""
    
    def __init__(self, name, description):
        """
        Initialize a concept.
        
        Args:
            name: The name of the concept
            description: A description of the concept
        """
        self.name = name
        self.description = description