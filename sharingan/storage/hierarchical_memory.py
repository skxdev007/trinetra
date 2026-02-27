"""
Hierarchical Memory Store for SHARINGAN Deep Architecture.

SYSTEM DESIGN OVERVIEW
=======================

The Hierarchical Memory Store is a three-level storage system that enables
multi-granularity video understanding and efficient query processing. Instead of
storing only raw frames or only high-level summaries, we maintain THREE levels
of representation:

1. **Frame-Level Memory**: Dense embeddings for every sampled frame
2. **Event-Level Memory**: Semantic events spanning multiple frames
3. **Chapter-Level Memory**: High-level narrative structure spanning multiple events

WHY HIERARCHICAL MEMORY?
-------------------------
Different queries require different levels of granularity:

- "What's in frame at 1:23?" → Frame-level lookup
- "Find person speaking" → Event-level semantic search
- "Summarize the video" → Chapter-level narrative retrieval

A flat storage system would be inefficient for all query types. The hierarchical
approach enables:

1. **Efficient Multi-Granularity Queries**: Route queries to appropriate level
2. **Memory Efficiency**: Store only necessary information at each level
3. **Referential Integrity**: Validate cross-level references (events → frames, chapters → events)
4. **Temporal Indexing**: Fast time-range queries at all levels

THREE-LEVEL ARCHITECTURE
-------------------------

Level 1: Frame Store
  - Stores: Frame descriptions, embeddings (INT8 compressed), timestamps
  - Granularity: Every sampled frame (1-5 FPS adaptive)
  - Size: ~2.3MB per 5-minute video (with INT8 quantization)
  - Use case: Fine-grained temporal queries, visual grounding

Level 2: Event Store
  - Stores: Event descriptions, embeddings, entity/action lists, frame_indices
  - Granularity: Semantic events (typically 2-10 seconds each)
  - Size: ~200 events per 5-minute video
  - Use case: Semantic search, causal reasoning, "what happened" queries

Level 3: Chapter Store
  - Stores: Chapter summaries, embeddings, key event IDs, time ranges
  - Granularity: High-level narrative segments (typically 30-120 seconds each)
  - Size: ~5-10 chapters per 5-minute video
  - Use case: Video summarization, navigation, high-level understanding

REFERENTIAL INTEGRITY
----------------------
The hierarchy enforces foreign key constraints:

- Events MUST reference valid frame_indices in frame_store
- Chapters MUST reference valid event_ids in event_store
- Adding invalid references raises ValueError

This ensures consistency and prevents dangling references.

QUERY ROUTING
-------------
The query_multi_level method automatically routes queries to appropriate levels:

- level="frame": Search only frame-level embeddings
- level="event": Search only event-level embeddings
- level="chapter": Search only chapter-level embeddings
- level="auto": Search all levels and combine results

Temporal window queries (query_temporal_window) work across all levels,
returning frames, events, and chapters within the specified time range.

MEMORY EFFICIENCY
-----------------
Frame embeddings use INT8 quantization for 4x memory reduction:
- Float32: 512 × 4 bytes = 2KB per frame
- INT8: 512 × 1 byte = 512 bytes per frame

For a 5-minute video at 1 FPS:
- 300 frames × 512 bytes = ~150KB (vs 600KB for Float32)

Event and chapter embeddings remain Float32 since they're much fewer in number.

HOW IT FITS IN THE SYSTEM
--------------------------
Ingest Pipeline:
  Video → Frames → SmolVLM → Frame Store
                           → Event Detection → Event Store
                                            → Chapter Segmentation → Chapter Store

Query Pipeline:
  User Query → Query Router → Memory Retrieval (multi-level)
                           → Reasoning Scaffold → LLM Response

The memory store is populated ONCE during ingest and queried FOREVER at O(1)
complexity (with indexed similarity search).

COMPLEXITY ANALYSIS
-------------------
- add_frame: O(1) - append to frame store
- add_event: O(F) - validate F frame_indices exist
- add_chapter: O(E) - validate E event_ids exist
- query_multi_level: O(N) - similarity search over N embeddings at target level
- query_temporal_window: O(F + E + C) - scan all three levels

In practice, F << T (300 frames from 9000 raw frames), E << F (200 events from 300 frames),
C << E (10 chapters from 200 events), so queries are fast.

EXAMPLE USAGE
-------------
```python
# Initialize hierarchical memory
memory = HierarchicalMemoryStore(cache_dir="cache/video_001")

# Add frame-level memory during ingest
for frame_desc, embedding in process_frames(video):
    memory.add_frame(
        frame_data=frame_desc,
        embedding=embedding
    )

# Add event-level memory after event detection
for event in detect_events(memory.frame_store):
    memory.add_event(
        event_data=event,
        frame_indices=[10, 11, 12, 13]  # Frames spanning this event
    )

# Add chapter-level memory after segmentation
for chapter in segment_chapters(memory.event_store):
    memory.add_chapter(
        chapter_data=chapter,
        event_ids=["evt_001", "evt_002", "evt_003"]
    )

# Query across all levels
query_embedding = encode_text("person speaking")
results = memory.query_multi_level(query_embedding, level="auto")

# Query temporal window
window_results = memory.query_temporal_window(start_time=10.0, end_time=20.0)
```

FUTURE ENHANCEMENTS (V2)
------------------------
- FAISS indexing for sub-linear similarity search on long videos
- Hierarchical clustering for automatic chapter detection
- Cross-level attention for joint reasoning
- Incremental updates for streaming video
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import json

from sharingan.storage.embedding_store import EmbeddingStore, QuantizationType


@dataclass
class FrameDescription:
    """
    Frame-level description with metadata.
    
    Attributes:
        timestamp: Time in seconds when frame occurs
        frame_index: Index of the frame in the video
        description: Natural language description of the frame
        entities: List of entities mentioned (objects, people, places)
        actions: List of actions/verbs in the frame
        confidence: Confidence score for the description (0.0 to 1.0)
        context_used: List of frame indices used as context for description
    """
    timestamp: float
    frame_index: int
    description: str
    entities: List[str]
    actions: List[str]
    confidence: float
    context_used: List[int]


@dataclass
class Event:
    """
    Event-level semantic moment spanning multiple frames.
    
    Attributes:
        event_id: Unique identifier for the event
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds
        description: Natural language description of the event
        entities: List of entities involved in the event
        actions: List of actions in the event
        frame_indices: List of frame indices that comprise this event
    """
    event_id: str
    start_time: float
    end_time: float
    description: str
    entities: List[str]
    actions: List[str]
    frame_indices: List[int]


@dataclass
class Chapter:
    """
    Chapter-level narrative segment spanning multiple events.
    
    Attributes:
        chapter_id: Unique identifier for the chapter
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds
        summary: High-level summary of the chapter
        key_events: List of event IDs that comprise this chapter
    """
    chapter_id: str
    start_time: float
    end_time: float
    summary: str
    key_events: List[str]


@dataclass
class MultiLevelResult:
    """
    Results from multi-level memory query.
    
    Attributes:
        frame_matches: List of matching frames with similarity scores
        event_matches: List of matching events with similarity scores
        chapter_matches: List of matching chapters with similarity scores
        reasoning_path: Explanation of how results were retrieved
    """
    frame_matches: List[tuple]  # [(FrameDescription, similarity_score), ...]
    event_matches: List[tuple]  # [(Event, similarity_score), ...]
    chapter_matches: List[tuple]  # [(Chapter, similarity_score), ...]
    reasoning_path: str


class FrameMemory:
    """
    Frame-level memory store with INT8 quantized embeddings.
    
    Stores dense frame descriptions and embeddings for fine-grained queries.
    Uses INT8 quantization for 4x memory reduction.
    """
    
    def __init__(self):
        """Initialize frame memory with INT8 quantization."""
        self.embedding_store = EmbeddingStore(quantization=QuantizationType.INT8)
        self.descriptions: List[FrameDescription] = []
    
    def add_frame(self, frame_data: FrameDescription, embedding: np.ndarray) -> None:
        """
        Add frame description and embedding.
        
        Args:
            frame_data: FrameDescription object with metadata
            embedding: Frame embedding vector (Float32)
        
        Raises:
            ValueError: If frame_index already exists
        """
        # Check for duplicate frame indices
        existing_indices = {desc.frame_index for desc in self.descriptions}
        if frame_data.frame_index in existing_indices:
            raise ValueError(
                f"Frame index {frame_data.frame_index} already exists in frame store"
            )
        
        # Add to embedding store
        self.embedding_store.add_embedding(
            embedding=embedding,
            timestamp=frame_data.timestamp,
            frame_index=frame_data.frame_index,
            metadata={
                "description": frame_data.description,
                "entities": frame_data.entities,
                "actions": frame_data.actions,
                "confidence": frame_data.confidence,
                "context_used": frame_data.context_used
            }
        )
        
        # Add to descriptions list
        self.descriptions.append(frame_data)
    
    def get_frame(self, frame_index: int) -> Optional[FrameDescription]:
        """
        Get frame description by index.
        
        Args:
            frame_index: Index of the frame to retrieve
        
        Returns:
            FrameDescription if found, None otherwise
        """
        for desc in self.descriptions:
            if desc.frame_index == frame_index:
                return desc
        return None
    
    def get_frames_in_window(
        self,
        start_time: float,
        end_time: float
    ) -> List[FrameDescription]:
        """
        Get all frames within a time window.
        
        Args:
            start_time: Start of time window (seconds)
            end_time: End of time window (seconds)
        
        Returns:
            List of FrameDescription objects within the time window
        """
        return [
            desc for desc in self.descriptions
            if start_time <= desc.timestamp <= end_time
        ]
    
    def count(self) -> int:
        """Return number of frames stored."""
        return len(self.descriptions)
    
    def __len__(self) -> int:
        """Return number of frames stored."""
        return len(self.descriptions)


class EventMemory:
    """
    Event-level memory store with semantic events.
    
    Stores events spanning multiple frames with entity/action information.
    Validates that all referenced frame_indices exist in frame_store.
    """
    
    def __init__(self):
        """Initialize event memory."""
        self.events: Dict[str, Event] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
    
    def add_event(
        self,
        event_data: Event,
        embedding: np.ndarray,
        frame_store: FrameMemory
    ) -> None:
        """
        Add event with validation of frame_indices.
        
        Args:
            event_data: Event object with metadata
            embedding: Event embedding vector (Float32)
            frame_store: FrameMemory to validate frame_indices against
        
        Raises:
            ValueError: If event_id already exists
            ValueError: If any frame_index doesn't exist in frame_store
        """
        # Check for duplicate event IDs
        if event_data.event_id in self.events:
            raise ValueError(
                f"Event {event_data.event_id} already exists in event store"
            )
        
        # Validate frame_indices exist in frame_store
        existing_frame_indices = {desc.frame_index for desc in frame_store.descriptions}
        for frame_idx in event_data.frame_indices:
            if frame_idx not in existing_frame_indices:
                raise ValueError(
                    f"Frame index {frame_idx} referenced by event {event_data.event_id} "
                    f"does not exist in frame_store"
                )
        
        # Add event
        self.events[event_data.event_id] = event_data
        self.embeddings[event_data.event_id] = embedding
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """
        Get event by ID.
        
        Args:
            event_id: ID of the event to retrieve
        
        Returns:
            Event if found, None otherwise
        """
        return self.events.get(event_id)
    
    def get_events_in_window(
        self,
        start_time: float,
        end_time: float
    ) -> List[Event]:
        """
        Get all events within a time window.
        
        Args:
            start_time: Start of time window (seconds)
            end_time: End of time window (seconds)
        
        Returns:
            List of Event objects that overlap with the time window
        """
        return [
            event for event in self.events.values()
            if not (event.end_time < start_time or event.start_time > end_time)
        ]
    
    def count(self) -> int:
        """Return number of events stored."""
        return len(self.events)
    
    def __len__(self) -> int:
        """Return number of events stored."""
        return len(self.events)


class ChapterMemory:
    """
    Chapter-level memory store with narrative structure.
    
    Stores high-level chapters spanning multiple events.
    Validates that all referenced event_ids exist in event_store.
    """
    
    def __init__(self):
        """Initialize chapter memory."""
        self.chapters: Dict[str, Chapter] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
    
    def add_chapter(
        self,
        chapter_data: Chapter,
        embedding: np.ndarray,
        event_store: EventMemory
    ) -> None:
        """
        Add chapter with validation of event_ids.
        
        Args:
            chapter_data: Chapter object with metadata
            embedding: Chapter embedding vector (Float32)
            event_store: EventMemory to validate event_ids against
        
        Raises:
            ValueError: If chapter_id already exists
            ValueError: If any event_id doesn't exist in event_store
        """
        # Check for duplicate chapter IDs
        if chapter_data.chapter_id in self.chapters:
            raise ValueError(
                f"Chapter {chapter_data.chapter_id} already exists in chapter store"
            )
        
        # Validate event_ids exist in event_store
        for event_id in chapter_data.key_events:
            if event_id not in event_store.events:
                raise ValueError(
                    f"Event ID {event_id} referenced by chapter {chapter_data.chapter_id} "
                    f"does not exist in event_store"
                )
        
        # Add chapter
        self.chapters[chapter_data.chapter_id] = chapter_data
        self.embeddings[chapter_data.chapter_id] = embedding
    
    def get_chapter(self, chapter_id: str) -> Optional[Chapter]:
        """
        Get chapter by ID.
        
        Args:
            chapter_id: ID of the chapter to retrieve
        
        Returns:
            Chapter if found, None otherwise
        """
        return self.chapters.get(chapter_id)
    
    def get_chapters_in_window(
        self,
        start_time: float,
        end_time: float
    ) -> List[Chapter]:
        """
        Get all chapters within a time window.
        
        Args:
            start_time: Start of time window (seconds)
            end_time: End of time window (seconds)
        
        Returns:
            List of Chapter objects that overlap with the time window
        """
        return [
            chapter for chapter in self.chapters.values()
            if not (chapter.end_time < start_time or chapter.start_time > end_time)
        ]
    
    def count(self) -> int:
        """Return number of chapters stored."""
        return len(self.chapters)
    
    def __len__(self) -> int:
        """Return number of chapters stored."""
        return len(self.chapters)


class HierarchicalMemoryStore:
    """
    Three-level hierarchical memory store for multi-granularity video understanding.
    
    Maintains three levels of representation:
    1. Frame-level: Dense embeddings for every sampled frame (INT8 compressed)
    2. Event-level: Semantic events spanning multiple frames
    3. Chapter-level: High-level narrative structure spanning multiple events
    
    Enforces referential integrity:
    - Events must reference valid frame_indices
    - Chapters must reference valid event_ids
    
    Supports multi-level queries and temporal window queries.
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize three-level memory hierarchy.
        
        Args:
            cache_dir: Directory path for caching memory to disk
        """
        self.frame_store = FrameMemory()
        self.event_store = EventMemory()
        self.chapter_store = ChapterMemory()
        self.cache_dir = Path(cache_dir)
    
    def add_frame(
        self,
        frame_data: FrameDescription,
        embedding: np.ndarray
    ) -> None:
        """
        Add frame-level memory.
        
        Args:
            frame_data: FrameDescription object with metadata
            embedding: Frame embedding vector (Float32, will be quantized to INT8)
        
        Raises:
            ValueError: If frame_index already exists
        """
        self.frame_store.add_frame(frame_data, embedding)
    
    def add_event(
        self,
        event_data: Event,
        frame_indices: List[int]
    ) -> None:
        """
        Add event-level memory with frame_indices validation.
        
        Args:
            event_data: Event object with metadata
            frame_indices: List of frame indices that comprise this event
        
        Raises:
            ValueError: If event_id already exists
            ValueError: If any frame_index doesn't exist in frame_store
        
        Note:
            This method computes the event embedding as the mean of frame embeddings.
            In a more sophisticated implementation, this could use a learned aggregation.
        """
        # Update event_data with frame_indices
        event_data.frame_indices = frame_indices
        
        # Compute event embedding as mean of frame embeddings
        frame_embeddings = []
        for frame_idx in frame_indices:
            # Find the embedding index for this frame_index
            for i, desc in enumerate(self.frame_store.descriptions):
                if desc.frame_index == frame_idx:
                    frame_emb = self.frame_store.embedding_store.get_embedding(i)
                    frame_embeddings.append(frame_emb)
                    break
        
        if not frame_embeddings:
            raise ValueError(
                f"No valid frame embeddings found for event {event_data.event_id}"
            )
        
        event_embedding = np.mean(frame_embeddings, axis=0)
        
        # Add to event store
        self.event_store.add_event(event_data, event_embedding, self.frame_store)
    
    def add_chapter(
        self,
        chapter_data: Chapter,
        event_ids: List[str]
    ) -> None:
        """
        Add chapter-level memory with event_ids validation.
        
        Args:
            chapter_data: Chapter object with metadata
            event_ids: List of event IDs that comprise this chapter
        
        Raises:
            ValueError: If chapter_id already exists
            ValueError: If any event_id doesn't exist in event_store
        
        Note:
            This method computes the chapter embedding as the mean of event embeddings.
        """
        # Update chapter_data with event_ids
        chapter_data.key_events = event_ids
        
        # Compute chapter embedding as mean of event embeddings
        event_embeddings = []
        for event_id in event_ids:
            if event_id in self.event_store.embeddings:
                event_embeddings.append(self.event_store.embeddings[event_id])
        
        if not event_embeddings:
            raise ValueError(
                f"No valid event embeddings found for chapter {chapter_data.chapter_id}"
            )
        
        chapter_embedding = np.mean(event_embeddings, axis=0)
        
        # Add to chapter store
        self.chapter_store.add_chapter(chapter_data, chapter_embedding, self.event_store)
    
    def query_multi_level(
        self,
        query_embedding: np.ndarray,
        level: str = "auto",
        top_k: int = 5
    ) -> MultiLevelResult:
        """
        Query across memory hierarchy with automatic level routing.
        
        Args:
            query_embedding: Query embedding vector (Float32)
            level: Memory level to query ("auto", "frame", "event", "chapter")
            top_k: Number of top results to return per level
        
        Returns:
            MultiLevelResult with matches from appropriate levels
        
        Raises:
            ValueError: If level is not valid
        
        Example:
            query_emb = encode_text("person speaking")
            results = memory.query_multi_level(query_emb, level="auto")
        """
        valid_levels = {"auto", "frame", "event", "chapter"}
        if level not in valid_levels:
            raise ValueError(f"Invalid level '{level}'. Must be one of {valid_levels}")
        
        frame_matches = []
        event_matches = []
        chapter_matches = []
        reasoning_path = f"Querying memory at level: {level}\n"
        
        # Query frame level
        if level in {"auto", "frame"}:
            if self.frame_store.count() > 0:
                frame_embeddings = self.frame_store.embedding_store.get_all_embeddings()
                similarities = self._compute_similarities(query_embedding, frame_embeddings)
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                for idx in top_indices:
                    frame_matches.append((
                        self.frame_store.descriptions[idx],
                        float(similarities[idx])
                    ))
                
                reasoning_path += f"Found {len(frame_matches)} frame matches\n"
        
        # Query event level
        if level in {"auto", "event"}:
            if self.event_store.count() > 0:
                event_ids = list(self.event_store.events.keys())
                event_embeddings = np.array([
                    self.event_store.embeddings[eid] for eid in event_ids
                ])
                similarities = self._compute_similarities(query_embedding, event_embeddings)
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                for idx in top_indices:
                    event_id = event_ids[idx]
                    event_matches.append((
                        self.event_store.events[event_id],
                        float(similarities[idx])
                    ))
                
                reasoning_path += f"Found {len(event_matches)} event matches\n"
        
        # Query chapter level
        if level in {"auto", "chapter"}:
            if self.chapter_store.count() > 0:
                chapter_ids = list(self.chapter_store.chapters.keys())
                chapter_embeddings = np.array([
                    self.chapter_store.embeddings[cid] for cid in chapter_ids
                ])
                similarities = self._compute_similarities(query_embedding, chapter_embeddings)
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                for idx in top_indices:
                    chapter_id = chapter_ids[idx]
                    chapter_matches.append((
                        self.chapter_store.chapters[chapter_id],
                        float(similarities[idx])
                    ))
                
                reasoning_path += f"Found {len(chapter_matches)} chapter matches\n"
        
        return MultiLevelResult(
            frame_matches=frame_matches,
            event_matches=event_matches,
            chapter_matches=chapter_matches,
            reasoning_path=reasoning_path
        )
    
    def query_temporal_window(
        self,
        start_time: float,
        end_time: float
    ) -> MultiLevelResult:
        """
        Retrieve data within time range across all three levels.
        
        Args:
            start_time: Start of time window (seconds)
            end_time: End of time window (seconds)
        
        Returns:
            MultiLevelResult with all frames, events, and chapters in the time window
        
        Raises:
            ValueError: If start_time > end_time
        
        Example:
            results = memory.query_temporal_window(10.0, 20.0)
        """
        if start_time > end_time:
            raise ValueError(
                f"Invalid time window: start_time ({start_time}) > end_time ({end_time})"
            )
        
        # Query all three levels
        frames = self.frame_store.get_frames_in_window(start_time, end_time)
        events = self.event_store.get_events_in_window(start_time, end_time)
        chapters = self.chapter_store.get_chapters_in_window(start_time, end_time)
        
        # Convert to match format (with dummy similarity scores)
        frame_matches = [(frame, 1.0) for frame in frames]
        event_matches = [(event, 1.0) for event in events]
        chapter_matches = [(chapter, 1.0) for chapter in chapters]
        
        reasoning_path = (
            f"Temporal window query: [{start_time:.1f}s, {end_time:.1f}s]\n"
            f"Found {len(frames)} frames, {len(events)} events, {len(chapters)} chapters"
        )
        
        return MultiLevelResult(
            frame_matches=frame_matches,
            event_matches=event_matches,
            chapter_matches=chapter_matches,
            reasoning_path=reasoning_path
        )
    
    def _compute_similarities(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarities between query and embeddings.
        
        Args:
            query_embedding: Query vector (D,)
            embeddings: Matrix of embeddings (N, D)
        
        Returns:
            Array of similarity scores (N,)
        """
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Normalize embeddings
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarities
        similarities = np.dot(embeddings_norm, query_norm)
        
        return similarities
    
    def __repr__(self) -> str:
        """String representation of the memory store."""
        return (
            f"HierarchicalMemoryStore("
            f"frames={self.frame_store.count()}, "
            f"events={self.event_store.count()}, "
            f"chapters={self.chapter_store.count()}"
            f")"
        )
