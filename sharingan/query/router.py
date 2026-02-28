"""
SYSTEM DESIGN: Query Router
============================

## What This Component Does

The Query Router is like a smart receptionist for your video questions. When you ask
a question about a video, the router figures out what KIND of question you're asking
and decides the best way to answer it.

Think of it like this: If you ask "What happened between 1:30 and 2:00?", the router
knows you want to see a specific time window. If you ask "Why did the car crash?",
it knows you're asking about CAUSE and EFFECT. Different questions need different
strategies to answer them well.

## Four Types of Questions

1. **Window Queries**: "What happened between 1:30 and 2:00?"
   - These ask about a specific time range
   - Router extracts the start and end times
   - System looks up frames/events in that time window

2. **Semantic Queries**: "Find the person speaking"
   - These ask about finding something based on meaning
   - Router extracts entities (person, car, dog, etc.)
   - System searches by similarity to find matching moments

3. **Causal Queries**: "Why did X happen?" or "What caused Y?"
   - These ask about cause and effect relationships
   - Router detects causal keywords (why, caused, because, led to)
   - System follows causal edges in the event graph

4. **Summary Queries**: "Summarize this video" or "What's the main story?"
   - These ask for a high-level overview
   - Router routes to chapter-level memory (not frame-by-frame)
   - System provides narrative summary

## How It Fits in the System

The Query Router is the FIRST step in answering any question:

1. User asks question → Query Router classifies it
2. Router creates a Query Plan (what type, what to look for, where to search)
3. Memory system retrieves relevant information based on the plan
4. Reasoning Scaffold structures the answer
5. Small LLM generates the final response

Without the router, we'd have to manually specify how to answer each question.
With the router, the system automatically picks the best strategy.

## Why This Matters

Small language models (0.5B parameters) struggle with complex reasoning. But if we
give them a STRUCTURED PLAN and the RIGHT CONTEXT, they can answer questions that
normally require much bigger models.

The router is what makes this possible - it breaks down complex questions into
simple, answerable pieces.

## Temporal Causality

The router respects temporal causality: it only routes to information that was
available at the time of the query. For causal queries, it ensures we only follow
edges from earlier events to later events (never backwards in time).

## Example

Query: "Why did the person pick up the knife?"

Router output:
- Type: causal
- Confidence: 0.95
- Causal keywords: ["why"]
- Entities: ["person", "knife"]
- Memory level: event
- Scaffold type: causal_chain
- Retrieval strategy: follow_causal_edges

This tells the system: "This is a causal question. Find events involving 'person'
and 'knife', follow causal edges between them, and format the answer as a chain
of cause-and-effect."
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re


@dataclass
class QueryType:
    """Classification result for a query.
    
    Attributes:
        type: One of "window", "semantic", "causal", "summary"
        confidence: How confident we are in this classification (0.0 to 1.0)
        temporal_bounds: For window queries, (start_time, end_time) in seconds
        entities: List of entities mentioned in the query (person, car, etc.)
        causal_keywords: List of causal keywords found (why, caused, because, etc.)
    """
    type: str
    confidence: float
    temporal_bounds: Optional[Tuple[float, float]] = None
    entities: List[str] = None
    causal_keywords: List[str] = None
    
    def __post_init__(self):
        """Initialize empty lists if None."""
        if self.entities is None:
            self.entities = []
        if self.causal_keywords is None:
            self.causal_keywords = []


@dataclass
class QueryPlan:
    """Execution plan for answering a query.
    
    Attributes:
        query_type: The classified query type
        memory_level: Which memory level to search ("frame", "event", "chapter")
        scaffold_type: Which reasoning template to use
        retrieval_strategy: How to retrieve relevant information
    """
    query_type: QueryType
    memory_level: str
    scaffold_type: str
    retrieval_strategy: str


class QueryRouter:
    """Routes natural language queries to appropriate handlers.
    
    The router analyzes incoming queries and determines:
    1. What type of question is being asked
    2. What information needs to be retrieved
    3. How to structure the reasoning process
    4. Which memory level to search
    
    This enables small LLMs to answer complex questions by providing
    structured guidance and relevant context.
    """
    
    # Causal keywords that indicate cause-and-effect questions
    CAUSAL_KEYWORDS = [
        "why", "caused", "because", "reason", "led to", "resulted in",
        "consequence", "effect", "due to", "thanks to", "trigger"
    ]
    
    # Summary keywords that indicate high-level overview requests
    SUMMARY_KEYWORDS = [
        "summarize", "summary", "overview", "main", "overall", "gist",
        "key points", "highlights", "recap", "brief"
    ]
    
    # Common entity types to extract
    ENTITY_PATTERNS = [
        r'\b(person|people|man|woman|child|boy|girl)\b',
        r'\b(car|vehicle|truck|bike|motorcycle)\b',
        r'\b(dog|cat|animal|pet)\b',
        r'\b(phone|computer|laptop|device)\b',
        r'\b(door|window|table|chair|furniture)\b',
        r'\b(food|drink|meal|coffee|water)\b',
    ]
    
    def __init__(self):
        """Initialize query router."""
        pass
    
    def classify_query(self, query: str) -> QueryType:
        """Classify query into one of four types.
        
        Args:
            query: Natural language query string
            
        Returns:
            QueryType with classification results
            
        Raises:
            ValueError: If query is empty or too long
        """
        if not query or len(query.strip()) == 0:
            raise ValueError("Query cannot be empty")
        
        if len(query) > 512:
            raise ValueError("Query too long (max 512 characters)")
        
        query_lower = query.lower().strip()
        
        # Check for temporal window queries first (most specific)
        temporal_bounds = self._extract_temporal_bounds(query_lower)
        if temporal_bounds is not None:
            entities = self._extract_entities(query_lower)
            return QueryType(
                type="window",
                confidence=0.95,
                temporal_bounds=temporal_bounds,
                entities=entities,
                causal_keywords=[]
            )
        
        # Check for causal queries
        causal_keywords = self._detect_causal_keywords(query_lower)
        if causal_keywords:
            entities = self._extract_entities(query_lower)
            return QueryType(
                type="causal",
                confidence=0.90,
                temporal_bounds=None,
                entities=entities,
                causal_keywords=causal_keywords
            )
        
        # Check for summary queries
        if self._is_summary_query(query_lower):
            return QueryType(
                type="summary",
                confidence=0.85,
                temporal_bounds=None,
                entities=[],
                causal_keywords=[]
            )
        
        # Default to semantic query
        entities = self._extract_entities(query_lower)
        return QueryType(
            type="semantic",
            confidence=0.80,
            temporal_bounds=None,
            entities=entities,
            causal_keywords=[]
        )
    
    def route_query(
        self,
        query: str,
        memory_store: Optional[any] = None
    ) -> QueryPlan:
        """Route query to appropriate handler with execution plan.
        
        Args:
            query: Natural language query string
            memory_store: Optional hierarchical memory store (for validation)
            
        Returns:
            QueryPlan with routing decisions
        """
        # Classify the query
        query_type = self.classify_query(query)
        
        # Determine memory level based on query type
        if query_type.type == "window":
            memory_level = "frame"  # Frame-level for precise time windows
        elif query_type.type == "semantic":
            memory_level = "event"  # Event-level for semantic search
        elif query_type.type == "causal":
            memory_level = "event"  # Event-level for causal reasoning
        elif query_type.type == "summary":
            memory_level = "chapter"  # Chapter-level for summaries
        else:
            memory_level = "auto"  # Let system decide
        
        # Determine scaffold type based on query type
        if query_type.type == "causal":
            scaffold_type = "causal_chain"
        elif query_type.type == "window":
            scaffold_type = "temporal_order"
        elif query_type.type == "summary":
            scaffold_type = "state_change"
        else:
            scaffold_type = "temporal_order"
        
        # Determine retrieval strategy
        if query_type.type == "window":
            retrieval_strategy = "temporal_window"
        elif query_type.type == "causal":
            retrieval_strategy = "follow_causal_edges"
        elif query_type.type == "summary":
            retrieval_strategy = "chapter_summary"
        else:
            retrieval_strategy = "semantic_similarity"
        
        return QueryPlan(
            query_type=query_type,
            memory_level=memory_level,
            scaffold_type=scaffold_type,
            retrieval_strategy=retrieval_strategy
        )
    
    def _extract_temporal_bounds(self, query: str) -> Optional[Tuple[float, float]]:
        """Extract temporal bounds from query.
        
        Looks for patterns like:
        - "between 1:30 and 2:00"
        - "from 0:45 to 1:15"
        - "at 2:30"
        
        Args:
            query: Lowercase query string
            
        Returns:
            Tuple of (start_time, end_time) in seconds, or None if not found
        """
        # Pattern: "between X and Y" or "from X to Y"
        pattern1 = r'(?:between|from)\s+(\d+):(\d+)\s+(?:and|to)\s+(\d+):(\d+)'
        match = re.search(pattern1, query)
        if match:
            start_min, start_sec, end_min, end_sec = map(int, match.groups())
            start_time = start_min * 60 + start_sec
            end_time = end_min * 60 + end_sec
            if start_time < end_time:
                return (start_time, end_time)
        
        # Pattern: "at X:Y" (single timestamp - use 5 second window)
        pattern2 = r'at\s+(\d+):(\d+)'
        match = re.search(pattern2, query)
        if match:
            minutes, seconds = map(int, match.groups())
            timestamp = minutes * 60 + seconds
            return (max(0, timestamp - 2.5), timestamp + 2.5)
        
        # Pattern: seconds only "between X and Y seconds"
        pattern3 = r'(?:between|from)\s+(\d+)\s+(?:and|to)\s+(\d+)\s+seconds?'
        match = re.search(pattern3, query)
        if match:
            start_time, end_time = map(int, match.groups())
            if start_time < end_time:
                return (start_time, end_time)
        
        return None
    
    def _detect_causal_keywords(self, query: str) -> List[str]:
        """Detect causal keywords in query.
        
        Args:
            query: Lowercase query string
            
        Returns:
            List of causal keywords found
        """
        found_keywords = []
        for keyword in self.CAUSAL_KEYWORDS:
            if keyword in query:
                found_keywords.append(keyword)
        return found_keywords
    
    def _is_summary_query(self, query: str) -> bool:
        """Check if query is asking for a summary.
        
        Args:
            query: Lowercase query string
            
        Returns:
            True if query contains summary keywords
        """
        for keyword in self.SUMMARY_KEYWORDS:
            if keyword in query:
                return True
        return False
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities mentioned in query.
        
        Args:
            query: Lowercase query string
            
        Returns:
            List of entities found
        """
        entities = []
        for pattern in self.ENTITY_PATTERNS:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities
