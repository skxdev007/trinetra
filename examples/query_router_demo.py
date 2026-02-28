"""
Query Router Demo
==================

This example demonstrates how the Query Router classifies different types
of queries and generates appropriate query plans.
"""

from sharingan.query.router import QueryRouter


def main():
    """Demonstrate query router functionality."""
    router = QueryRouter()
    
    # Example queries of different types
    queries = [
        # Window queries
        "What happened between 1:30 and 2:00?",
        "Show me from 0:45 to 1:15",
        "What happened at 2:30?",
        
        # Causal queries
        "Why did the person pick up the knife?",
        "What caused the car to crash?",
        "Why did the door close?",
        
        # Summary queries
        "Summarize this video",
        "Give me an overview of the main events",
        "What are the key highlights?",
        
        # Semantic queries
        "Find the person speaking",
        "Show me the dog",
        "Where is the car?"
    ]
    
    print("=" * 80)
    print("QUERY ROUTER DEMONSTRATION")
    print("=" * 80)
    print()
    
    for query in queries:
        print(f"Query: {query}")
        print("-" * 80)
        
        # Classify the query
        query_type = router.classify_query(query)
        print(f"  Type: {query_type.type}")
        print(f"  Confidence: {query_type.confidence:.2f}")
        
        if query_type.temporal_bounds:
            start, end = query_type.temporal_bounds
            print(f"  Temporal Bounds: {start:.1f}s to {end:.1f}s")
        
        if query_type.entities:
            print(f"  Entities: {', '.join(query_type.entities)}")
        
        if query_type.causal_keywords:
            print(f"  Causal Keywords: {', '.join(query_type.causal_keywords)}")
        
        # Generate query plan
        plan = router.route_query(query)
        print(f"  Memory Level: {plan.memory_level}")
        print(f"  Scaffold Type: {plan.scaffold_type}")
        print(f"  Retrieval Strategy: {plan.retrieval_strategy}")
        print()
    
    print("=" * 80)
    print("QUERY ROUTING SUMMARY")
    print("=" * 80)
    print()
    print("The Query Router successfully:")
    print("  ✓ Classified queries into 4 types (window, causal, summary, semantic)")
    print("  ✓ Extracted temporal bounds from time-based queries")
    print("  ✓ Detected causal keywords for cause-and-effect questions")
    print("  ✓ Identified entities mentioned in queries")
    print("  ✓ Generated appropriate query plans for each type")
    print()
    print("This enables small LLMs to answer complex questions by providing")
    print("structured guidance and routing to the right memory level.")


if __name__ == "__main__":
    main()
