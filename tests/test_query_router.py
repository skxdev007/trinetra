"""Tests for Query Router."""

import pytest
from sharingan.query.router import QueryRouter, QueryType, QueryPlan


class TestQueryRouter:
    """Test suite for QueryRouter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.router = QueryRouter()
    
    def test_window_query_classification(self):
        """Test classification of temporal window queries."""
        query = "What happened between 1:30 and 2:00?"
        result = self.router.classify_query(query)
        
        assert result.type == "window"
        assert result.confidence >= 0.9
        assert result.temporal_bounds is not None
        assert result.temporal_bounds[0] == 90  # 1:30 in seconds
        assert result.temporal_bounds[1] == 120  # 2:00 in seconds
    
    def test_window_query_from_to(self):
        """Test window query with 'from X to Y' pattern."""
        query = "Show me from 0:45 to 1:15"
        result = self.router.classify_query(query)
        
        assert result.type == "window"
        assert result.temporal_bounds == (45, 75)
    
    def test_window_query_at_timestamp(self):
        """Test window query with single timestamp."""
        query = "What happened at 2:30?"
        result = self.router.classify_query(query)
        
        assert result.type == "window"
        assert result.temporal_bounds is not None
        # Should create a 5-second window around the timestamp
        assert result.temporal_bounds[0] == 147.5  # 2:30 - 2.5s
        assert result.temporal_bounds[1] == 152.5  # 2:30 + 2.5s
    
    def test_causal_query_why(self):
        """Test causal query with 'why' keyword."""
        query = "Why did the person pick up the knife?"
        result = self.router.classify_query(query)
        
        assert result.type == "causal"
        assert result.confidence >= 0.85
        assert "why" in result.causal_keywords
        assert "person" in result.entities
    
    def test_causal_query_caused(self):
        """Test causal query with 'caused' keyword."""
        query = "What caused the car to crash?"
        result = self.router.classify_query(query)
        
        assert result.type == "causal"
        assert "caused" in result.causal_keywords
        assert "car" in result.entities
    
    def test_summary_query(self):
        """Test summary query classification."""
        query = "Summarize this video"
        result = self.router.classify_query(query)
        
        assert result.type == "summary"
        assert result.confidence >= 0.8
        assert result.temporal_bounds is None
    
    def test_summary_query_overview(self):
        """Test summary query with 'overview' keyword."""
        query = "Give me an overview of the main events"
        result = self.router.classify_query(query)
        
        assert result.type == "summary"
    
    def test_semantic_query(self):
        """Test semantic query classification (default)."""
        query = "Find the person speaking"
        result = self.router.classify_query(query)
        
        assert result.type == "semantic"
        assert "person" in result.entities
        assert len(result.causal_keywords) == 0
    
    def test_entity_extraction(self):
        """Test entity extraction from queries."""
        query = "Show me the person with the dog and the car"
        result = self.router.classify_query(query)
        
        assert "person" in result.entities
        assert "dog" in result.entities
        assert "car" in result.entities
    
    def test_empty_query_raises_error(self):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            self.router.classify_query("")
    
    def test_long_query_raises_error(self):
        """Test that overly long query raises ValueError."""
        long_query = "a" * 513
        with pytest.raises(ValueError, match="Query too long"):
            self.router.classify_query(long_query)
    
    def test_route_window_query(self):
        """Test routing of window query."""
        query = "What happened between 1:00 and 2:00?"
        plan = self.router.route_query(query)
        
        assert plan.query_type.type == "window"
        assert plan.memory_level == "frame"
        assert plan.scaffold_type == "temporal_order"
        assert plan.retrieval_strategy == "temporal_window"
    
    def test_route_causal_query(self):
        """Test routing of causal query."""
        query = "Why did the person leave?"
        plan = self.router.route_query(query)
        
        assert plan.query_type.type == "causal"
        assert plan.memory_level == "event"
        assert plan.scaffold_type == "causal_chain"
        assert plan.retrieval_strategy == "follow_causal_edges"
    
    def test_route_summary_query(self):
        """Test routing of summary query."""
        query = "Summarize the video"
        plan = self.router.route_query(query)
        
        assert plan.query_type.type == "summary"
        assert plan.memory_level == "chapter"
        assert plan.scaffold_type == "state_change"
        assert plan.retrieval_strategy == "chapter_summary"
    
    def test_route_semantic_query(self):
        """Test routing of semantic query."""
        query = "Find the dog"
        plan = self.router.route_query(query)
        
        assert plan.query_type.type == "semantic"
        assert plan.memory_level == "event"
        assert plan.scaffold_type == "temporal_order"
        assert plan.retrieval_strategy == "semantic_similarity"
    
    def test_confidence_bounds(self):
        """Test that confidence scores are within valid bounds."""
        queries = [
            "What happened between 1:00 and 2:00?",
            "Why did this happen?",
            "Summarize the video",
            "Find the person"
        ]
        
        for query in queries:
            result = self.router.classify_query(query)
            assert 0.0 <= result.confidence <= 1.0
    
    def test_temporal_bounds_ordering(self):
        """Test that temporal bounds are properly ordered."""
        query = "What happened between 2:00 and 1:00?"
        result = self.router.classify_query(query)
        
        # Should not extract bounds if end < start
        # Falls back to semantic query
        assert result.type == "semantic"
    
    def test_seconds_pattern(self):
        """Test temporal extraction with seconds pattern."""
        query = "Show me between 30 and 60 seconds"
        result = self.router.classify_query(query)
        
        assert result.type == "window"
        assert result.temporal_bounds == (30, 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
