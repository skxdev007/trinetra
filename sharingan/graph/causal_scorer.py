"""
Causal Edge Scorer for SHARINGAN Deep Architecture

SYSTEM DESIGN OVERVIEW
=======================

The Causal Edge Scorer is a critical component for building the Temporal Event Graph.
It determines the type and strength of relationships between events, enabling causal
reasoning queries like "Why did X happen?" and "What caused Y?".

WHY CAUSAL EDGE SCORING?
-------------------------
Traditional video understanding treats events independently or uses simple temporal
adjacency. This fails for causal reasoning because:

1. **Not all temporal adjacency implies causality**
   - "Person walks" followed by "Car drives by" are temporally adjacent but not causal
   - "Person picks up knife" followed by "Person cuts vegetables" are causally related

2. **Semantic similarity alone is insufficient**
   - "Cooking pasta" and "Boiling water" are semantically similar but one causes the other
   - "Person enters room" and "Person exits room" are semantically similar but opposite

3. **Time delta is a strong causal signal**
   - Events 0.5-2s apart are more likely causal than events 30s apart
   - But time alone doesn't capture semantic causality

The Causal Edge Scorer combines multiple signals to classify edges as:
- **Causal**: Cause-effect relationships (for "why" queries)
- **Semantic**: Thematic relationships (for "find related" queries)
- **Temporal**: Simple adjacency (for "what happened next" queries)

V1 vs V2 IMPLEMENTATION
-----------------------

**V1 (Current - Heuristic Mode)**:
- Uses cosine similarity between event embeddings as primary signal
- Applies fixed thresholds: >0.7 = causal, 0.5-0.7 = semantic, <0.5 = temporal
- Incorporates time delta encoding for temporal awareness
- Fast, no training required, sufficient for initial paper submission
- Limitations: Fixed thresholds may not generalize across video domains

**V2 (Future - Learned Mode)**:
- Trains a neural network on NExT-QA and TemporalBench datasets
- Learns to predict edge types from event embeddings + time delta
- Architecture: MLP with feature extraction (concat, diff, product) + time encoding
- Expected improvement: 10-15% better edge classification accuracy
- Timeline: 3-4 weeks additional work (dataset prep, training, evaluation)

For the initial paper submission, V1 heuristic mode is sufficient to demonstrate
the system's temporal reasoning capabilities. V2 can be added in a follow-up paper
or extended version.

EDGE SCORING ALGORITHM
----------------------

The scorer processes two events and their time delta:

1. **Feature Extraction**:
   - Concatenation: [emb1; emb2] - captures both events
   - Difference: emb2 - emb1 - captures change/transition
   - Product: emb1 * emb2 - captures interaction
   - Time encoding: sinusoidal encoding of time delta

2. **V1 Heuristic Scoring**:
   - Compute cosine similarity between embeddings
   - Apply thresholds:
     * similarity > 0.7 → causal edge (high semantic overlap + temporal proximity)
     * 0.5 ≤ similarity ≤ 0.7 → semantic edge (moderate overlap)
     * similarity < 0.5 → temporal edge (low overlap, just temporal adjacency)
   - Adjust confidence based on time delta (closer events = higher confidence)

3. **V2 Learned Scoring** (future):
   - Pass combined features through MLP: features → hidden → logits
   - Apply softmax to get probabilities for [causal, semantic, temporal]
   - Return edge type with highest probability

CONFIDENCE SCORE ADJUSTMENT
----------------------------
The confidence score is adjusted based on time delta:
- 0-2s: Full confidence (immediate causality)
- 2-10s: 90% confidence (likely causality)
- 10-30s: 70% confidence (possible causality)
- >30s: 50% confidence (weak causality)

This reflects the intuition that causal relationships are stronger when events
are temporally close.

HOW IT FITS IN THE SYSTEM
--------------------------

Ingest Pipeline:
  Video → Frames → SmolVLM → Events → **Causal Edge Scorer** → Event Graph → Memory

The scorer is called during event graph construction:
1. For each pair of events (e1, e2) where e1.timestamp < e2.timestamp
2. Score the potential edge between them
3. If confidence > 0.5, add edge to graph
4. Result: Fully connected event graph with typed edges

Query Pipeline:
  User Query → Router → Graph Traversal (uses edge types) → Scaffold → LLM

The edge types enable different query strategies:
- Causal queries: Traverse only causal edges (find_causal_chain)
- Semantic queries: Traverse semantic edges (find related events)
- Temporal queries: Use temporal edges (what happened next)

COMPLEXITY ANALYSIS
-------------------
- score_edge: O(D) where D = embedding dimension (typically 512)
  * Cosine similarity: O(D)
  * Time encoding: O(D)
  * Feature extraction: O(D)

- Building full graph: O(E²) where E = number of events
  * Must score all pairs of events
  * In practice, E << T (200 events from 100K frames)
  * For 200 events: 200² = 40K edge scorings (fast on GPU)

While O(E²) sounds expensive, it's actually very fast because:
1. E is small (100-500 events for typical videos)
2. Each scoring is just vector operations (no neural network in V1)
3. Can be parallelized on GPU
4. Only done once during ingest, not during queries

CONFIGURATION
-------------
The scorer supports two modes via configuration flag:
- mode="heuristic": Use V1 cosine similarity thresholds (default)
- mode="learned": Use V2 trained neural network (requires trained model)

This allows easy switching between V1 and V2 without changing the interface.

EXAMPLE USAGE
-------------
```python
# Initialize scorer in heuristic mode (V1)
scorer = CausalEdgeScorer(embedding_dim=512, mode="heuristic")

# Score edge between two events
event1 = graph.get_event("evt_001")  # "Person picks up knife"
event2 = graph.get_event("evt_002")  # "Person cuts vegetables"

edge_score = scorer.score_edge(
    event1_embedding=event1.embedding,
    event2_embedding=event2.embedding,
    time_delta=event2.timestamp - event1.timestamp
)

print(f"Edge type: {edge_score.edge_type}")        # "causal"
print(f"Confidence: {edge_score.confidence:.2f}")  # 0.85
print(f"Reasoning: {edge_score.reasoning}")        # "High semantic similarity..."

# Add edge to graph if confidence is sufficient
if edge_score.confidence > 0.5:
    graph.add_edge(
        source_id=event1.event_id,
        target_id=event2.event_id,
        edge_type=edge_score.edge_type,
        confidence=edge_score.confidence
    )
```

FUTURE ENHANCEMENTS (V2)
------------------------
- Train on NExT-QA causal/temporal/descriptive question annotations
- Use graph neural networks for multi-hop reasoning
- Incorporate visual features (not just embeddings)
- Probabilistic edge scoring with uncertainty quantification
- Active learning to improve scorer with user feedback
"""

from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
import torch
import torch.nn as nn

from sharingan.temporal.time_encoding import ContinuousTimeEncoder


@dataclass
class EdgeScore:
    """
    Result of scoring a potential edge between two events.
    
    Attributes:
        edge_type: Type of edge ("causal", "semantic", "temporal")
        confidence: Confidence score in [0.0, 1.0]
        time_delta: Time difference between events in seconds
        reasoning: Human-readable explanation of the edge classification
    """
    edge_type: Literal["causal", "semantic", "temporal"]
    confidence: float
    time_delta: float
    reasoning: str


class CausalEdgeScorer(nn.Module):
    """
    Scores causal relationships between events for graph construction.
    
    This scorer determines the type and strength of edges in the Temporal Event Graph.
    It supports two modes:
    
    1. **Heuristic Mode (V1)**: Uses cosine similarity thresholds
       - Fast, no training required
       - Sufficient for initial paper submission
       - Thresholds: >0.7 = causal, 0.5-0.7 = semantic, <0.5 = temporal
    
    2. **Learned Mode (V2)**: Uses trained neural network
       - Requires training on NExT-QA dataset
       - Better accuracy (expected 10-15% improvement)
       - Future enhancement (3-4 weeks additional work)
    
    The scorer combines multiple signals:
    - Event embedding similarity (semantic relatedness)
    - Time delta encoding (temporal proximity)
    - Feature interactions (concatenation, difference, product)
    
    Complexity: O(D) per edge where D = embedding dimension
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        num_edge_types: int = 3,  # causal, semantic, temporal
        mode: Literal["heuristic", "learned"] = "heuristic",
        device: str = "cpu"
    ):
        """
        Initialize causal edge scorer.
        
        Args:
            embedding_dim: Dimension of event embeddings (default 512)
            hidden_dim: Hidden layer dimension for learned mode (default 256)
            num_edge_types: Number of edge types to classify (default 3)
            mode: Scoring mode - "heuristic" (V1) or "learned" (V2)
            device: Device for computation ("cpu" or "cuda")
        
        Raises:
            ValueError: If mode is not "heuristic" or "learned"
        """
        super().__init__()
        
        if mode not in ["heuristic", "learned"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'heuristic' or 'learned'")
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        self.mode = mode
        self.device = device
        
        # Initialize time encoder
        self.time_encoder = ContinuousTimeEncoder(d_model=embedding_dim)
        
        # V2 Learned Mode: Neural network for edge classification
        # Feature dimensions:
        # - Concatenation: 2 * embedding_dim
        # - Difference: embedding_dim
        # - Product: embedding_dim
        # - Time encoding: embedding_dim
        # Total: 5 * embedding_dim
        if mode == "learned":
            feature_dim = 5 * embedding_dim
            
            self.feature_network = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_edge_types)
            )
            
            self.to(device)
        
        # V1 Heuristic Mode: Fixed thresholds
        self.causal_threshold = 0.7
        self.semantic_threshold = 0.5
    
    def _extract_features(
        self,
        event1_embedding: np.ndarray,
        event2_embedding: np.ndarray,
        time_delta: float
    ) -> np.ndarray:
        """
        Extract interaction features from event embeddings and time delta.
        
        This method computes multiple feature representations that capture
        different aspects of the relationship between events:
        
        1. Concatenation [emb1; emb2]: Preserves both events independently
        2. Difference (emb2 - emb1): Captures transition/change
        3. Product (emb1 * emb2): Captures element-wise interaction
        4. Time encoding: Captures temporal distance
        
        Args:
            event1_embedding: Embedding of first event (shape: embedding_dim)
            event2_embedding: Embedding of second event (shape: embedding_dim)
            time_delta: Time difference in seconds
        
        Returns:
            Combined feature vector of shape (5 * embedding_dim,)
        """
        # Ensure embeddings are numpy arrays
        emb1 = np.asarray(event1_embedding, dtype=np.float32)
        emb2 = np.asarray(event2_embedding, dtype=np.float32)
        
        # Validate shapes
        if emb1.shape != (self.embedding_dim,):
            raise ValueError(
                f"event1_embedding must have shape ({self.embedding_dim},), "
                f"got {emb1.shape}"
            )
        if emb2.shape != (self.embedding_dim,):
            raise ValueError(
                f"event2_embedding must have shape ({self.embedding_dim},), "
                f"got {emb2.shape}"
            )
        
        # Feature 1: Concatenation
        concat_features = np.concatenate([emb1, emb2])
        
        # Feature 2: Difference (captures change/transition)
        diff_features = emb2 - emb1
        
        # Feature 3: Element-wise product (captures interaction)
        product_features = emb1 * emb2
        
        # Feature 4: Time delta encoding
        time_encoding = self.time_encoder.encode_time_delta(time_delta)
        
        # Combine all features
        combined_features = np.concatenate([
            concat_features,
            diff_features,
            product_features,
            time_encoding
        ])
        
        return combined_features
    
    def _compute_cosine_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
        
        Returns:
            Cosine similarity in [-1.0, 1.0]
        """
        # Normalize embeddings
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        
        # Compute dot product
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Clip to valid range (numerical stability)
        similarity = np.clip(similarity, -1.0, 1.0)
        
        return float(similarity)
    
    def _adjust_confidence_by_time(
        self,
        base_confidence: float,
        time_delta: float
    ) -> float:
        """
        Adjust confidence score based on time delta.
        
        Causal relationships are stronger when events are temporally close.
        This method applies a decay function to reduce confidence for distant events.
        
        Time delta ranges:
        - 0-2s: Full confidence (immediate causality)
        - 2-10s: 90% confidence (likely causality)
        - 10-30s: 70% confidence (possible causality)
        - >30s: 50% confidence (weak causality)
        
        Args:
            base_confidence: Initial confidence score
            time_delta: Time difference in seconds
        
        Returns:
            Adjusted confidence score in [0.0, 1.0]
        """
        if time_delta <= 2.0:
            time_factor = 1.0
        elif time_delta <= 10.0:
            time_factor = 0.9
        elif time_delta <= 30.0:
            time_factor = 0.7
        else:
            time_factor = 0.5
        
        adjusted_confidence = base_confidence * time_factor
        
        return float(np.clip(adjusted_confidence, 0.0, 1.0))
    
    def _score_heuristic(
        self,
        event1_embedding: np.ndarray,
        event2_embedding: np.ndarray,
        time_delta: float
    ) -> EdgeScore:
        """
        Score edge using V1 heuristic mode (cosine similarity thresholds).
        
        This method uses fixed thresholds on cosine similarity to classify edges:
        - similarity > 0.7: Causal edge (high semantic overlap)
        - 0.5 ≤ similarity ≤ 0.7: Semantic edge (moderate overlap)
        - similarity < 0.5: Temporal edge (low overlap)
        
        The confidence is adjusted based on time delta to reflect that closer
        events are more likely to be causally related.
        
        Args:
            event1_embedding: Embedding of first event
            event2_embedding: Embedding of second event
            time_delta: Time difference in seconds
        
        Returns:
            EdgeScore with edge_type, confidence, time_delta, and reasoning
        """
        # Compute cosine similarity
        similarity = self._compute_cosine_similarity(
            event1_embedding,
            event2_embedding
        )
        
        # Classify edge type based on thresholds
        if similarity > self.causal_threshold:
            edge_type = "causal"
            base_confidence = similarity
            reasoning = (
                f"High semantic similarity ({similarity:.3f}) suggests causal relationship. "
                f"Events are {time_delta:.1f}s apart."
            )
        elif similarity >= self.semantic_threshold:
            edge_type = "semantic"
            base_confidence = similarity
            reasoning = (
                f"Moderate semantic similarity ({similarity:.3f}) suggests thematic relationship. "
                f"Events are {time_delta:.1f}s apart."
            )
        else:
            edge_type = "temporal"
            base_confidence = 0.5  # Low confidence for temporal-only edges
            reasoning = (
                f"Low semantic similarity ({similarity:.3f}) suggests temporal adjacency only. "
                f"Events are {time_delta:.1f}s apart."
            )
        
        # Adjust confidence based on time delta
        confidence = self._adjust_confidence_by_time(base_confidence, time_delta)
        
        return EdgeScore(
            edge_type=edge_type,
            confidence=confidence,
            time_delta=time_delta,
            reasoning=reasoning
        )
    
    def _score_learned(
        self,
        event1_embedding: np.ndarray,
        event2_embedding: np.ndarray,
        time_delta: float
    ) -> EdgeScore:
        """
        Score edge using V2 learned mode (neural network).
        
        This method uses a trained neural network to classify edges based on
        extracted features (concatenation, difference, product, time encoding).
        
        Args:
            event1_embedding: Embedding of first event
            event2_embedding: Embedding of second event
            time_delta: Time difference in seconds
        
        Returns:
            EdgeScore with edge_type, confidence, time_delta, and reasoning
        
        Raises:
            RuntimeError: If called in heuristic mode (network not initialized)
        """
        if self.mode != "learned":
            raise RuntimeError("_score_learned called but mode is not 'learned'")
        
        # Extract features
        features = self._extract_features(
            event1_embedding,
            event2_embedding,
            time_delta
        )
        
        # Convert to tensor
        features_tensor = torch.from_numpy(features).float().to(self.device)
        features_tensor = features_tensor.unsqueeze(0)  # Add batch dimension
        
        # Forward pass through network
        with torch.no_grad():
            logits = self.feature_network(features_tensor)
            probs = torch.softmax(logits, dim=-1)
        
        # Get predicted edge type and confidence
        probs_np = probs.cpu().numpy()[0]
        edge_type_idx = int(np.argmax(probs_np))
        confidence = float(probs_np[edge_type_idx])
        
        edge_types = ["causal", "semantic", "temporal"]
        edge_type = edge_types[edge_type_idx]
        
        # Generate reasoning
        reasoning = (
            f"Neural network classified as {edge_type} with confidence {confidence:.3f}. "
            f"Events are {time_delta:.1f}s apart."
        )
        
        return EdgeScore(
            edge_type=edge_type,
            confidence=confidence,
            time_delta=time_delta,
            reasoning=reasoning
        )
    
    def score_edge(
        self,
        event1_embedding: np.ndarray,
        event2_embedding: np.ndarray,
        time_delta: float
    ) -> EdgeScore:
        """
        Score potential edge between two events.
        
        This is the main entry point for edge scoring. It dispatches to either
        heuristic or learned mode based on the scorer's configuration.
        
        Args:
            event1_embedding: Embedding of first event (earlier in time)
            event2_embedding: Embedding of second event (later in time)
            time_delta: Time difference in seconds (must be positive)
        
        Returns:
            EdgeScore containing edge_type, confidence, time_delta, and reasoning
        
        Raises:
            ValueError: If time_delta is not positive
            ValueError: If embeddings have incorrect shape
        
        Example:
            >>> scorer = CausalEdgeScorer(mode="heuristic")
            >>> edge_score = scorer.score_edge(
            ...     event1_embedding=np.random.randn(512),
            ...     event2_embedding=np.random.randn(512),
            ...     time_delta=2.5
            ... )
            >>> print(edge_score.edge_type)  # "causal", "semantic", or "temporal"
            >>> print(edge_score.confidence)  # 0.0 to 1.0
        """
        # Validate time delta
        if time_delta <= 0:
            raise ValueError(f"time_delta must be positive, got {time_delta}")
        
        # Dispatch to appropriate scoring method
        if self.mode == "heuristic":
            return self._score_heuristic(
                event1_embedding,
                event2_embedding,
                time_delta
            )
        else:  # mode == "learned"
            return self._score_learned(
                event1_embedding,
                event2_embedding,
                time_delta
            )
    
    def train_on_qa_data(
        self,
        qa_dataset,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 32
    ) -> None:
        """
        Train scorer on video QA dataset (V2 learned mode only).
        
        This method trains the neural network on annotated video QA data
        (e.g., NExT-QA, TemporalBench) to learn causal edge patterns.
        
        NOTE: This is a placeholder for V2 implementation. Full training
        pipeline will be implemented in Phase 9.
        
        Args:
            qa_dataset: Video QA dataset with causal annotations
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
        
        Raises:
            RuntimeError: If called in heuristic mode
            NotImplementedError: V2 training not yet implemented
        """
        if self.mode != "learned":
            raise RuntimeError("train_on_qa_data requires mode='learned'")
        
        raise NotImplementedError(
            "V2 training pipeline not yet implemented. "
            "This will be added in Phase 9 (Causal Reasoning V2). "
            "For now, use mode='heuristic' for edge scoring."
        )
    
    def __repr__(self) -> str:
        return (
            f"CausalEdgeScorer("
            f"mode={self.mode}, "
            f"embedding_dim={self.embedding_dim}, "
            f"device={self.device}"
            f")"
        )
