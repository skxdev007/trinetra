"""
Query intent classification for temporal video QA.

Classifies queries into types (point, comparative, counting, causal, boundary)
and extracts temporal constraints for intelligent retrieval routing.

Author: SHARINGAN Team
Date: March 6, 2026
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import List


class QueryType(Enum):
    """Types of video queries."""
    POINT = "point"              # "Find X", "When did X happen"
    COMPARATIVE = "comparative"  # "First X vs last X", "Compare X in beginning and end"
    COUNTING = "counting"        # "How many X", "Count X"
    CAUSAL = "causal"           # "Why did X happen", "What caused X"
    TEMPORAL_BOUNDARY = "boundary"  # "What happens at the end", "Beginning of video"


@dataclass
class TemporalConstraint:
    """Temporal constraint extracted from query."""
    type: str  # "first", "last", "beginning", "end", "middle", "early", "late"
    window_start: float  # Percentage of video (0.0 to 1.0)
    window_end: float    # Percentage of video (0.0 to 1.0)


@dataclass
class QueryIntent:
    """Parsed query intent."""
    query_type: QueryType
    constraints: List[TemporalConstraint]
    keywords: List[str]
    requires_dual_window: bool


class QueryIntentClassifier:
    """Classify query intent and extract temporal constraints."""
    
    # Patterns for query types
    COMPARATIVE_PATTERNS = [
        r'\b(first|beginning|early|initial)\b.*\b(vs|versus|compared to|and)\b.*\b(last|end|final|late)\b',
        r'\b(last|end|final|late)\b.*\b(vs|versus|compared to|and)\b.*\b(first|beginning|early|initial)\b',
        r'\bcompare\b.*\b(first|beginning)\b.*\b(last|end)\b',
        r'\bcompare\b.*\b(last|end)\b.*\b(first|beginning)\b',
        r'\bdifference between\b.*\b(first|beginning)\b.*\b(last|end)\b',
    ]
    
    COUNTING_PATTERNS = [
        r'\bhow many\b',
        r'\bcount\b',
        r'\bnumber of\b',
        r'\bhow often\b',
    ]
    
    CAUSAL_PATTERNS = [
        r'\bwhy\b',
        r'\bwhat caused\b',
        r'\breason for\b',
        r'\bexplain\b',
    ]
    
    BOUNDARY_PATTERNS = [
        r'\bat the (beginning|start|end|finish)\b',
        r'\b(beginning|start|end|finish) of\b',
    ]
    
    # Temporal keywords and their windows
    TEMPORAL_WINDOWS = {
        'first': (0.0, 0.2),      # First 20%
        'beginning': (0.0, 0.15),  # First 15%
        'early': (0.0, 0.25),      # First 25%
        'initial': (0.0, 0.2),     # First 20%
        'start': (0.0, 0.1),       # First 10%
        
        'last': (0.8, 1.0),        # Last 20%
        'end': (0.85, 1.0),        # Last 15%
        'final': (0.9, 1.0),       # Last 10%
        'late': (0.75, 1.0),       # Last 25%
        'finish': (0.9, 1.0),      # Last 10%
        
        'middle': (0.4, 0.6),      # Middle 20%
    }
    
    def classify(self, query: str) -> QueryIntent:
        """
        Classify query intent and extract temporal constraints.
        
        Args:
            query: User query string
            
        Returns:
            QueryIntent with type, constraints, and metadata
            
        Example:
            >>> classifier = QueryIntentClassifier()
            >>> intent = classifier.classify("Compare repairs in first vs last project")
            >>> print(intent.query_type)
            QueryType.COMPARATIVE
            >>> print(intent.requires_dual_window)
            True
        """
        query_lower = query.lower()
        
        # Detect query type
        query_type = self._detect_query_type(query_lower)
        
        # Extract temporal constraints
        constraints = self._extract_temporal_constraints(query_lower)
        
        # Extract keywords
        keywords = self._extract_keywords(query_lower)
        
        # Check if dual-window search needed
        requires_dual_window = (
            query_type == QueryType.COMPARATIVE and 
            len(constraints) >= 2
        )
        
        return QueryIntent(
            query_type=query_type,
            constraints=constraints,
            keywords=keywords,
            requires_dual_window=requires_dual_window
        )
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect query type from patterns."""
        # Check comparative first (most specific)
        for pattern in self.COMPARATIVE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.COMPARATIVE
        
        # Check counting
        for pattern in self.COUNTING_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.COUNTING
        
        # Check causal
        for pattern in self.CAUSAL_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.CAUSAL
        
        # Check boundary
        for pattern in self.BOUNDARY_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.TEMPORAL_BOUNDARY
        
        # Default to point lookup
        return QueryType.POINT
    
    def _extract_temporal_constraints(self, query: str) -> List[TemporalConstraint]:
        """Extract temporal constraints from query."""
        constraints = []
        
        for keyword, (start, end) in self.TEMPORAL_WINDOWS.items():
            if re.search(rf'\b{keyword}\b', query):
                constraints.append(TemporalConstraint(
                    type=keyword,
                    window_start=start,
                    window_end=end
                ))
        
        return constraints
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Remove stop words and temporal keywords
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'vs', 'versus'}
        temporal_words = set(self.TEMPORAL_WINDOWS.keys())
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and w not in temporal_words]
        
        return keywords


def test_intent_classifier():
    """Test query intent classifier."""
    
    print("Testing Query Intent Classifier...")
    print("=" * 60)
    
    classifier = QueryIntentClassifier()
    
    # Test 1: Comparative query
    print("\nTest 1: Comparative Query")
    print("-" * 60)
    
    query = "Compare repairs in first vs last project"
    intent = classifier.classify(query)
    
    print(f"Query: '{query}'")
    print(f"Type: {intent.query_type}")
    print(f"Constraints: {len(intent.constraints)}")
    for c in intent.constraints:
        print(f"  - {c.type}: {c.window_start:.1%} to {c.window_end:.1%}")
    print(f"Requires dual-window: {intent.requires_dual_window}")
    
    assert intent.query_type == QueryType.COMPARATIVE
    assert intent.requires_dual_window == True
    assert len(intent.constraints) >= 2
    print("✓ Comparative query detected correctly")
    
    # Test 2: Point query
    print("\nTest 2: Point Query")
    print("-" * 60)
    
    query = "When is epoxy used?"
    intent = classifier.classify(query)
    
    print(f"Query: '{query}'")
    print(f"Type: {intent.query_type}")
    print(f"Requires dual-window: {intent.requires_dual_window}")
    
    assert intent.query_type == QueryType.POINT
    assert intent.requires_dual_window == False
    print("✓ Point query detected correctly")
    
    # Test 3: Counting query
    print("\nTest 3: Counting Query")
    print("-" * 60)
    
    query = "How many times was epoxy used?"
    intent = classifier.classify(query)
    
    print(f"Query: '{query}'")
    print(f"Type: {intent.query_type}")
    
    assert intent.query_type == QueryType.COUNTING
    print("✓ Counting query detected correctly")
    
    # Test 4: Temporal boundary query
    print("\nTest 4: Temporal Boundary Query")
    print("-" * 60)
    
    query = "What happens at the end?"
    intent = classifier.classify(query)
    
    print(f"Query: '{query}'")
    print(f"Type: {intent.query_type}")
    print(f"Constraints: {len(intent.constraints)}")
    for c in intent.constraints:
        print(f"  - {c.type}: {c.window_start:.1%} to {c.window_end:.1%}")
    
    assert intent.query_type == QueryType.TEMPORAL_BOUNDARY
    print("✓ Temporal boundary query detected correctly")
    
    print("\n" + "=" * 60)
    print("All tests passed!")


if __name__ == "__main__":
    test_intent_classifier()
