"""
Temporal Event Graph for SHARINGAN Deep Architecture.

SYSTEM DESIGN OVERVIEW
=======================

The Temporal Event Graph is a core component of SHARINGAN's causal reasoning system.
It represents video understanding as a directed graph where:
- Nodes = Events (semantic moments with timestamp, description, entities, actions)
- Edges = Relationships (causal, semantic, or temporal connections between events)

WHY TEMPORAL EVENT GRAPHS?
---------------------------
Traditional video understanding treats frames independently or uses simple temporal
pooling. This fails for causal reasoning questions like "Why did X happen?" or
"What caused Y?". The Temporal Event Graph enables:

1. **Causal Chain Discovery**: Find sequences of causally-related events
   - Example: "Person picks up knife" → "Person cuts vegetables" → "Person cooks meal"
   
2. **Temporal Window Queries**: Retrieve all events within a time range
   - Example: "What happened between 1:30 and 2:00?"
   
3. **Multi-Scale Reasoning**: Connect events at different temporal granularities
   - Short-term: gesture-level causality (2-4 frames)
   - Mid-term: action-level causality (8-16 frames)
   - Long-term: scene-level causality (32+ frames)

EDGE TYPES
----------
The graph supports three edge types, each serving a different reasoning purpose:

1. **Causal Edges** (edge_type="causal"):
   - Represent cause-effect relationships
   - Used for "why" and "what caused" queries
   - Example: "Picking up knife" causes "Cutting vegetables"
   - Scored by CausalEdgeScorer (heuristic or learned)

2. **Semantic Edges** (edge_type="semantic"):
   - Represent thematic or conceptual relationships
   - Used for "find related events" queries
   - Example: "Cooking pasta" relates to "Boiling water"
   - Based on embedding similarity

3. **Temporal Edges** (edge_type="temporal"):
   - Represent simple temporal adjacency
   - Used for "what happened next" queries
   - Example: "Person enters room" then "Person sits down"
   - Based on timestamp proximity

TEMPORAL ORDERING CONSTRAINT
-----------------------------
The graph enforces strict temporal causality:
- All edges must go from earlier events to later events
- source.timestamp < target.timestamp is ALWAYS validated
- This prevents logical inconsistencies and future information leakage

GRAPH TRAVERSAL ALGORITHMS
---------------------------
1. **find_causal_chain**: BFS on causal edges only
   - Finds shortest causal path between two events
   - Returns empty list if no causal path exists
   - Used for causal reasoning queries

2. **query_temporal_window**: Time-range filtering
   - Returns all events within [start_time, end_time]
   - Used for temporal window queries
   - Efficient O(N) scan with timestamp filtering

HOW IT FITS IN THE SYSTEM
--------------------------
Ingest Pipeline:
  Video → Frames → SmolVLM Descriptions → Event Detection → Event Graph Construction
  
Query Pipeline:
  User Query → Query Router → Graph Traversal → Reasoning Scaffold → LLM Response

The event graph is built ONCE during ingest (O(E²) for edge scoring where E = events)
and queried FOREVER at O(1) or O(E) depending on query type.

COMPLEXITY ANALYSIS
-------------------
- add_event: O(1) - simple dictionary insertion
- add_edge: O(1) - append to edge list with validation
- find_causal_chain: O(E + C) where E = events, C = causal edges (BFS)
- query_temporal_window: O(E) - linear scan with timestamp filtering

In practice, E << T (200 events from 100K frames), so graph operations are fast.

EXAMPLE USAGE
-------------
```python
# Build event graph during ingest
graph = TemporalEventGraph()

# Add events from frame descriptions
event1_id = graph.add_event(
    event_id="evt_001",
    timestamp=10.5,
    description="Person picks up knife",
    embedding=np.array([...]),
    entities=["person", "knife"],
    actions=["pick_up"]
)

event2_id = graph.add_event(
    event_id="evt_002",
    timestamp=15.2,
    description="Person cuts vegetables",
    embedding=np.array([...]),
    entities=["person", "vegetables", "knife"],
    actions=["cut"]
)

# Add causal edge (scored by CausalEdgeScorer)
graph.add_edge(
    source_id="evt_001",
    target_id="evt_002",
    edge_type="causal",
    confidence=0.85
)

# Query causal chain
chain = graph.find_causal_chain("evt_001", "evt_002")
# Returns: [EventNode(evt_001), EventNode(evt_002)]

# Query temporal window
events = graph.query_temporal_window(10.0, 20.0)
# Returns: [EventNode(evt_001), EventNode(evt_002)]
```

FUTURE ENHANCEMENTS (V2)
------------------------
- Learned edge scoring using NExT-QA dataset
- Graph neural networks for multi-hop reasoning
- Hierarchical event clustering for chapter-level structure
- Probabilistic causal inference with uncertainty quantification
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import deque
import numpy as np


@dataclass
class EventNode:
    """
    Represents a semantic event in the video.
    
    An event is a meaningful moment with:
    - Unique identifier and timestamp
    - Natural language description
    - Embedding vector for similarity search
    - Extracted entities (objects, people, places)
    - Extracted actions (verbs, activities)
    
    Events are detected from frame descriptions using temporal segmentation.
    """
    event_id: str
    timestamp: float
    description: str
    embedding: np.ndarray
    entities: List[str]
    actions: List[str]


@dataclass
class EventEdge:
    """
    Represents a directed relationship between two events.
    
    Edge types:
    - "causal": cause-effect relationship (A causes B)
    - "semantic": thematic relationship (A relates to B)
    - "temporal": temporal adjacency (A happens before B)
    
    Confidence score indicates edge strength (0.0 to 1.0).
    Time delta is the temporal distance between events (in seconds).
    """
    source_id: str
    target_id: str
    edge_type: str  # "causal", "semantic", "temporal"
    confidence: float
    time_delta: float


class TemporalEventGraph:
    """
    Temporal Event Graph for causal reasoning and temporal queries.
    
    This graph structure enables:
    1. Causal chain discovery (find_causal_chain)
    2. Temporal window queries (query_temporal_window)
    3. Multi-scale temporal reasoning
    
    The graph enforces strict temporal ordering:
    - All edges go from earlier to later events
    - source.timestamp < target.timestamp is always validated
    
    Complexity:
    - add_event: O(1)
    - add_edge: O(1) with validation
    - find_causal_chain: O(E + C) where E = events, C = causal edges
    - query_temporal_window: O(E)
    """
    
    def __init__(self):
        """Initialize empty event graph."""
        self.nodes: Dict[str, EventNode] = {}
        self.edges: List[EventEdge] = []
        
        # Build adjacency list for efficient graph traversal
        # Format: {source_id: [(target_id, edge_type, confidence), ...]}
        self._adjacency: Dict[str, List[tuple]] = {}
    
    def add_event(
        self,
        event_id: str,
        timestamp: float,
        description: str,
        embedding: np.ndarray,
        entities: List[str],
        actions: List[str]
    ) -> str:
        """
        Add event node to graph.
        
        Args:
            event_id: Unique identifier for the event
            timestamp: Time in seconds when event occurs
            description: Natural language description of the event
            embedding: Vector embedding for similarity search (typically 512-dim)
            entities: List of entities mentioned (objects, people, places)
            actions: List of actions/verbs in the event
        
        Returns:
            event_id: The ID of the added event (same as input)
        
        Raises:
            ValueError: If event_id already exists in the graph
        """
        if event_id in self.nodes:
            raise ValueError(f"Event {event_id} already exists in graph")
        
        node = EventNode(
            event_id=event_id,
            timestamp=timestamp,
            description=description,
            embedding=embedding,
            entities=entities,
            actions=actions
        )
        
        self.nodes[event_id] = node
        self._adjacency[event_id] = []  # Initialize empty adjacency list
        
        return event_id
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        confidence: float
    ) -> None:
        """
        Add edge between events with temporal ordering validation.
        
        Args:
            source_id: ID of source event (earlier in time)
            target_id: ID of target event (later in time)
            edge_type: Type of edge ("causal", "semantic", "temporal")
            confidence: Confidence score for the edge (0.0 to 1.0)
        
        Raises:
            ValueError: If source or target event doesn't exist
            ValueError: If edge_type is not valid
            ValueError: If temporal ordering is violated (source >= target)
            ValueError: If confidence is not in [0.0, 1.0]
        """
        # Validate event existence
        if source_id not in self.nodes:
            raise ValueError(f"Source event {source_id} not found in graph")
        if target_id not in self.nodes:
            raise ValueError(f"Target event {target_id} not found in graph")
        
        # Validate edge type
        valid_edge_types = {"causal", "semantic", "temporal"}
        if edge_type not in valid_edge_types:
            raise ValueError(
                f"Invalid edge_type '{edge_type}'. Must be one of {valid_edge_types}"
            )
        
        # Validate confidence score
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {confidence}")
        
        # CRITICAL: Validate temporal ordering
        source_node = self.nodes[source_id]
        target_node = self.nodes[target_id]
        
        if source_node.timestamp >= target_node.timestamp:
            raise ValueError(
                f"Temporal ordering violation: source timestamp "
                f"({source_node.timestamp}) must be < target timestamp "
                f"({target_node.timestamp})"
            )
        
        # Compute time delta
        time_delta = target_node.timestamp - source_node.timestamp
        
        # Create edge
        edge = EventEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            confidence=confidence,
            time_delta=time_delta
        )
        
        self.edges.append(edge)
        
        # Update adjacency list for efficient traversal
        self._adjacency[source_id].append((target_id, edge_type, confidence))
    
    def find_causal_chain(
        self,
        start_event: str,
        end_event: str
    ) -> List[EventNode]:
        """
        Find shortest causal path between two events using BFS.
        
        This method traverses ONLY causal edges to find cause-effect chains.
        It's used for answering "why" and "what caused" queries.
        
        Args:
            start_event: ID of starting event
            end_event: ID of ending event
        
        Returns:
            List of EventNode objects representing the causal chain from
            start_event to end_event. Returns empty list if no causal path exists.
        
        Raises:
            ValueError: If start_event or end_event doesn't exist
        
        Example:
            chain = graph.find_causal_chain("evt_001", "evt_003")
            # Returns: [EventNode(evt_001), EventNode(evt_002), EventNode(evt_003)]
            # Representing: evt_001 → evt_002 → evt_003
        """
        # Validate event existence
        if start_event not in self.nodes:
            raise ValueError(f"Start event {start_event} not found in graph")
        if end_event not in self.nodes:
            raise ValueError(f"End event {end_event} not found in graph")
        
        # BFS to find shortest causal path
        queue = deque([(start_event, [start_event])])
        visited = {start_event}
        
        while queue:
            current_id, path = queue.popleft()
            
            # Check if we reached the target
            if current_id == end_event:
                # Convert event IDs to EventNode objects
                return [self.nodes[event_id] for event_id in path]
            
            # Explore neighbors via causal edges only
            for neighbor_id, edge_type, confidence in self._adjacency[current_id]:
                if edge_type == "causal" and neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
        
        # No causal path found
        return []
    
    def query_temporal_window(
        self,
        start_time: float,
        end_time: float
    ) -> List[EventNode]:
        """
        Query events within a time range.
        
        Returns all events where start_time <= event.timestamp <= end_time.
        Results are sorted by timestamp in ascending order.
        
        Args:
            start_time: Start of time window (seconds)
            end_time: End of time window (seconds)
        
        Returns:
            List of EventNode objects within the time window, sorted by timestamp
        
        Raises:
            ValueError: If start_time > end_time
        
        Example:
            events = graph.query_temporal_window(10.0, 20.0)
            # Returns all events between 10s and 20s
        """
        if start_time > end_time:
            raise ValueError(
                f"Invalid time window: start_time ({start_time}) > end_time ({end_time})"
            )
        
        # Filter events by timestamp
        events_in_window = [
            node for node in self.nodes.values()
            if start_time <= node.timestamp <= end_time
        ]
        
        # Sort by timestamp
        events_in_window.sort(key=lambda node: node.timestamp)
        
        return events_in_window
    
    def get_event(self, event_id: str) -> Optional[EventNode]:
        """
        Get event node by ID.
        
        Args:
            event_id: ID of the event to retrieve
        
        Returns:
            EventNode if found, None otherwise
        """
        return self.nodes.get(event_id)
    
    def get_edges_from(self, event_id: str) -> List[EventEdge]:
        """
        Get all edges originating from an event.
        
        Args:
            event_id: ID of the source event
        
        Returns:
            List of EventEdge objects where source_id == event_id
        """
        return [edge for edge in self.edges if edge.source_id == event_id]
    
    def get_edges_to(self, event_id: str) -> List[EventEdge]:
        """
        Get all edges pointing to an event.
        
        Args:
            event_id: ID of the target event
        
        Returns:
            List of EventEdge objects where target_id == event_id
        """
        return [edge for edge in self.edges if edge.target_id == event_id]
    
    def get_causal_edges(self) -> List[EventEdge]:
        """
        Get all causal edges in the graph.
        
        Returns:
            List of EventEdge objects where edge_type == "causal"
        """
        return [edge for edge in self.edges if edge.edge_type == "causal"]
    
    def __len__(self) -> int:
        """Return number of events in the graph."""
        return len(self.nodes)
    
    def __repr__(self) -> str:
        """String representation of the graph."""
        num_causal = len([e for e in self.edges if e.edge_type == "causal"])
        num_semantic = len([e for e in self.edges if e.edge_type == "semantic"])
        num_temporal = len([e for e in self.edges if e.edge_type == "temporal"])
        
        return (
            f"TemporalEventGraph("
            f"events={len(self.nodes)}, "
            f"edges={len(self.edges)} "
            f"[causal={num_causal}, semantic={num_semantic}, temporal={num_temporal}]"
            f")"
        )
