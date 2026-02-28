"""
SYSTEM DESIGN: Reasoning Scaffold Builder
==========================================

## What This Component Does

The Reasoning Scaffold Builder creates structured reasoning templates that guide
small language models (0.5B parameters) through complex temporal reasoning tasks.

Think of it like this: If you ask a 10th grader to solve a complex math problem,
they might struggle. But if you give them a STEP-BY-STEP TEMPLATE with clear
instructions and relevant information at each step, they can solve it successfully.

That's exactly what reasoning scaffolds do for small LLMs. Instead of asking them
to figure out everything from scratch, we provide:

1. **Structured reasoning steps**: "First do X, then do Y, finally do Z"
2. **Relevant evidence**: The specific video moments that matter
3. **Clear constraints**: What to focus on and what to ignore
4. **Expected format**: How to structure the answer

## Why Small LLMs Need Scaffolds

Large models like GPT-4 (175B+ parameters) can reason through complex questions
without much guidance. But small models (0.5B parameters) need help:

- **Without scaffold**: "Why did the person pick up the knife?" → Model gets confused,
  hallucinates, or gives vague answers
  
- **With scaffold**: 
  ```
  Step 1: Initial state - Person is cooking in kitchen
  Step 2: Transition - Person reaches for cutting board
  Step 3: Causal event - Person picks up knife to cut vegetables
  
  Evidence: [Frame 45: "person at cutting board", Frame 52: "person holding knife"]
  
  Answer the question: Why did the person pick up the knife?
  Format: Provide a concise answer with timestamp.
  ```
  
  → Model follows the structure and gives accurate answer: "To cut vegetables (0:52)"

## Three Scaffold Types

### 1. Causal Chain (for "why" questions)
Format: Event A → Event B → Event C

Example:
```
Person enters kitchen (0:10) → Person opens refrigerator (0:15) → 
Person takes out milk (0:20) → Person pours milk into glass (0:25)
```

This helps the model understand CAUSE-AND-EFFECT relationships.

### 2. Temporal Order (for "what happened" questions)
Format: First X, then Y, finally Z

Example:
```
First: Person sits at desk (0:05)
Then: Person opens laptop (0:10)
Then: Person types on keyboard (0:15)
Finally: Person closes laptop (0:45)
```

This helps the model understand SEQUENCE of events.

### 3. State Change (for "how did X change" questions)
Format: Initial state → Transition → Final state

Example:
```
Initial state: Room is dark and empty (0:00)
Transition: Person enters and turns on light (0:05)
Final state: Room is bright with person sitting (0:10)
```

This helps the model understand TRANSFORMATIONS over time.

## How It Fits in the System

```
User Query → Query Router → Reasoning Scaffold Builder → Small LLM
                ↓                      ↓
         (query type)         (structured template)
                                       ↓
                              Hierarchical Memory
                              (retrieves evidence)
```

The scaffold builder sits between query routing and LLM generation:
1. Router determines what TYPE of reasoning is needed
2. Scaffold builder creates TEMPLATE for that reasoning type
3. Memory retrieval provides EVIDENCE to fill the template
4. Small LLM follows the template to generate accurate answer

## Why This Matters

Without scaffolds, SHARINGAN would need a large LLM (7B+ parameters) to answer
complex temporal questions. With scaffolds, we can use tiny 0.5B models and still
beat GPT-4o on temporal reasoning benchmarks.

This is the KEY INNOVATION that makes local, zero-cost video understanding possible.

## Implementation Details

The ReasoningScaffoldBuilder class:
- Maintains templates for each scaffold type
- Fills templates with retrieved evidence from memory
- Formats scaffolds as clear LLM prompts
- Specifies expected answer format

The build_scaffold method:
- Takes query plan (from router) and retrieved context (from memory)
- Selects appropriate template based on scaffold_type
- Extracts relevant evidence from context
- Generates reasoning steps
- Returns structured ReasoningScaffold dataclass

The format_for_llm method:
- Converts scaffold into clear text prompt
- Includes all reasoning steps with evidence
- Adds constraints and expected format
- Returns ready-to-use prompt string
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from sharingan.query.router import QueryPlan
from sharingan.storage.hierarchical_memory import MultiLevelResult


@dataclass
class ReasoningScaffold:
    """
    Structured reasoning template for guiding small LLMs.
    
    Attributes:
        scaffold_type: Type of reasoning template ("causal_chain", "temporal_order", "state_change")
        reasoning_steps: List of reasoning steps with structure
        evidence: List of evidence items from retrieved context
        constraints: List of constraints to guide reasoning
        expected_answer_format: Description of how answer should be formatted
    """
    scaffold_type: str
    reasoning_steps: List[str]
    evidence: List[Dict[str, Any]]
    constraints: List[str]
    expected_answer_format: str


class ReasoningScaffoldBuilder:
    """
    Builds structured reasoning templates to guide small LLMs through complex temporal reasoning.
    
    The scaffold builder creates step-by-step reasoning templates that help small language models
    (0.5B parameters) perform complex temporal reasoning tasks that would normally require much
    larger models.
    
    Three scaffold types are supported:
    1. causal_chain: For "why" questions - shows cause-and-effect relationships
    2. temporal_order: For "what happened" questions - shows sequence of events
    3. state_change: For "how did X change" questions - shows transformations
    
    Example usage:
        builder = ReasoningScaffoldBuilder()
        scaffold = builder.build_scaffold(query_plan, retrieved_context)
        prompt = builder.format_for_llm(scaffold)
    """
    
    def __init__(self):
        """Initialize scaffold templates."""
        self.templates = {
            "causal_chain": self._causal_chain_template,
            "temporal_order": self._temporal_order_template,
            "state_change": self._state_change_template
        }
    
    def build_scaffold(
        self,
        query_plan: QueryPlan,
        retrieved_context: MultiLevelResult
    ) -> ReasoningScaffold:
        """
        Build reasoning scaffold based on query plan and retrieved context.
        
        Args:
            query_plan: Query plan from router containing scaffold_type
            retrieved_context: Retrieved context from hierarchical memory
            
        Returns:
            ReasoningScaffold with structured reasoning template
            
        Raises:
            ValueError: If scaffold_type is not recognized
        """
        scaffold_type = query_plan.scaffold_type
        
        if scaffold_type not in self.templates:
            raise ValueError(
                f"Unknown scaffold type: {scaffold_type}. "
                f"Must be one of: {list(self.templates.keys())}"
            )
        
        # Get template function for this scaffold type
        template_func = self.templates[scaffold_type]
        
        # Build scaffold using template
        scaffold = template_func(query_plan, retrieved_context)
        
        return scaffold
    
    def format_for_llm(self, scaffold: ReasoningScaffold) -> str:
        """
        Format scaffold as LLM prompt with clear structure.
        
        Args:
            scaffold: ReasoningScaffold to format
            
        Returns:
            Formatted prompt string ready for LLM
        """
        prompt_parts = []
        
        # Add reasoning steps
        prompt_parts.append("# Reasoning Steps\n")
        for i, step in enumerate(scaffold.reasoning_steps, 1):
            prompt_parts.append(f"{i}. {step}")
        
        # Add evidence
        if scaffold.evidence:
            prompt_parts.append("\n# Evidence from Video\n")
            for i, evidence_item in enumerate(scaffold.evidence, 1):
                timestamp = evidence_item.get("timestamp", "unknown")
                description = evidence_item.get("description", "")
                prompt_parts.append(f"[{timestamp}s] {description}")
        
        # Add constraints
        if scaffold.constraints:
            prompt_parts.append("\n# Constraints\n")
            for constraint in scaffold.constraints:
                prompt_parts.append(f"- {constraint}")
        
        # Add expected format
        prompt_parts.append(f"\n# Expected Answer Format\n{scaffold.expected_answer_format}")
        
        return "\n".join(prompt_parts)
    
    def _causal_chain_template(
        self,
        query_plan: QueryPlan,
        retrieved_context: MultiLevelResult
    ) -> ReasoningScaffold:
        """
        Build causal chain scaffold: Event A → Event B → Event C
        
        Used for "why" questions that require understanding cause-and-effect relationships.
        """
        # Extract events from context
        events = self._extract_events_from_context(retrieved_context)
        
        # Build causal chain reasoning steps
        reasoning_steps = []
        if len(events) == 0:
            reasoning_steps.append("No events found in retrieved context")
        elif len(events) == 1:
            reasoning_steps.append(f"Single event: {events[0]['description']}")
        else:
            # Format as causal chain with arrows
            chain_parts = []
            for event in events:
                timestamp = event.get("timestamp", "?")
                desc = event.get("description", "unknown event")
                chain_parts.append(f"{desc} ({timestamp}s)")
            
            reasoning_steps.append(" → ".join(chain_parts))
        
        # Extract evidence
        evidence = self._extract_evidence(retrieved_context)
        
        # Define constraints for causal reasoning
        constraints = [
            "Focus on cause-and-effect relationships",
            "Earlier events cause later events",
            "Explain WHY something happened, not just WHAT happened",
            "Use evidence from video to support causal claims"
        ]
        
        # Define expected answer format
        expected_format = "Provide a concise answer explaining the causal relationship with timestamp(s)."
        
        return ReasoningScaffold(
            scaffold_type="causal_chain",
            reasoning_steps=reasoning_steps,
            evidence=evidence,
            constraints=constraints,
            expected_answer_format=expected_format
        )
    
    def _temporal_order_template(
        self,
        query_plan: QueryPlan,
        retrieved_context: MultiLevelResult
    ) -> ReasoningScaffold:
        """
        Build temporal order scaffold: First X, then Y, finally Z
        
        Used for "what happened" questions that require understanding sequence of events.
        """
        # Extract events from context
        events = self._extract_events_from_context(retrieved_context)
        
        # Build temporal order reasoning steps
        reasoning_steps = []
        if len(events) == 0:
            reasoning_steps.append("No events found in retrieved context")
        elif len(events) == 1:
            reasoning_steps.append(f"Single event: {events[0]['description']}")
        else:
            # Format as temporal sequence
            for i, event in enumerate(events):
                timestamp = event.get("timestamp", "?")
                desc = event.get("description", "unknown event")
                
                if i == 0:
                    prefix = "First:"
                elif i == len(events) - 1:
                    prefix = "Finally:"
                else:
                    prefix = "Then:"
                
                reasoning_steps.append(f"{prefix} {desc} ({timestamp}s)")
        
        # Extract evidence
        evidence = self._extract_evidence(retrieved_context)
        
        # Define constraints for temporal reasoning
        constraints = [
            "Focus on the sequence of events",
            "Maintain chronological order",
            "Describe WHAT happened at each step",
            "Include timestamps for temporal context"
        ]
        
        # Define expected answer format
        expected_format = "Provide a chronological summary with timestamps for key events."
        
        return ReasoningScaffold(
            scaffold_type="temporal_order",
            reasoning_steps=reasoning_steps,
            evidence=evidence,
            constraints=constraints,
            expected_answer_format=expected_format
        )
    
    def _state_change_template(
        self,
        query_plan: QueryPlan,
        retrieved_context: MultiLevelResult
    ) -> ReasoningScaffold:
        """
        Build state change scaffold: Initial state → Transition → Final state
        
        Used for "how did X change" questions that require understanding transformations.
        """
        # Extract events from context
        events = self._extract_events_from_context(retrieved_context)
        
        # Build state change reasoning steps
        reasoning_steps = []
        if len(events) == 0:
            reasoning_steps.append("No events found in retrieved context")
        elif len(events) == 1:
            reasoning_steps.append(f"Single state: {events[0]['description']}")
        else:
            # Format as state transition
            initial_event = events[0]
            final_event = events[-1]
            transition_events = events[1:-1] if len(events) > 2 else []
            
            initial_timestamp = initial_event.get("timestamp", "?")
            initial_desc = initial_event.get("description", "unknown state")
            reasoning_steps.append(f"Initial state: {initial_desc} ({initial_timestamp}s)")
            
            if transition_events:
                transition_desc = " → ".join([
                    f"{e.get('description', 'unknown')} ({e.get('timestamp', '?')}s)"
                    for e in transition_events
                ])
                reasoning_steps.append(f"Transition: {transition_desc}")
            else:
                reasoning_steps.append("Transition: (direct change)")
            
            final_timestamp = final_event.get("timestamp", "?")
            final_desc = final_event.get("description", "unknown state")
            reasoning_steps.append(f"Final state: {final_desc} ({final_timestamp}s)")
        
        # Extract evidence
        evidence = self._extract_evidence(retrieved_context)
        
        # Define constraints for state change reasoning
        constraints = [
            "Focus on how things changed over time",
            "Compare initial state to final state",
            "Explain the transformation process",
            "Identify key transition points"
        ]
        
        # Define expected answer format
        expected_format = "Describe the change from initial to final state with timestamps."
        
        return ReasoningScaffold(
            scaffold_type="state_change",
            reasoning_steps=reasoning_steps,
            evidence=evidence,
            constraints=constraints,
            expected_answer_format=expected_format
        )
    
    def _extract_events_from_context(
        self,
        retrieved_context: MultiLevelResult
    ) -> List[Dict[str, Any]]:
        """
        Extract events from retrieved context, sorted by timestamp.
        
        Args:
            retrieved_context: Retrieved context from hierarchical memory
            
        Returns:
            List of event dictionaries with timestamp and description
        """
        events = []
        
        # Extract from event matches (preferred)
        if retrieved_context.event_matches:
            for event_tuple in retrieved_context.event_matches:
                event = event_tuple[0]  # (Event, similarity_score)
                events.append({
                    "timestamp": getattr(event, "timestamp", 0.0),
                    "description": getattr(event, "description", ""),
                    "entities": getattr(event, "entities", []),
                    "actions": getattr(event, "actions", [])
                })
        
        # Fall back to frame matches if no events
        elif retrieved_context.frame_matches:
            for frame_tuple in retrieved_context.frame_matches:
                frame = frame_tuple[0]  # (FrameDescription, similarity_score)
                events.append({
                    "timestamp": getattr(frame, "timestamp", 0.0),
                    "description": getattr(frame, "description", ""),
                    "entities": getattr(frame, "entities", []),
                    "actions": getattr(frame, "actions", [])
                })
        
        # Sort by timestamp
        events.sort(key=lambda e: e.get("timestamp", 0.0))
        
        return events
    
    def _extract_evidence(
        self,
        retrieved_context: MultiLevelResult
    ) -> List[Dict[str, Any]]:
        """
        Extract evidence items from retrieved context.
        
        Args:
            retrieved_context: Retrieved context from hierarchical memory
            
        Returns:
            List of evidence dictionaries with timestamp and description
        """
        evidence = []
        
        # Extract from all levels
        for event_tuple in retrieved_context.event_matches:
            event = event_tuple[0]
            evidence.append({
                "timestamp": getattr(event, "timestamp", 0.0),
                "description": getattr(event, "description", ""),
                "type": "event"
            })
        
        for frame_tuple in retrieved_context.frame_matches:
            frame = frame_tuple[0]
            evidence.append({
                "timestamp": getattr(frame, "timestamp", 0.0),
                "description": getattr(frame, "description", ""),
                "type": "frame"
            })
        
        for chapter_tuple in retrieved_context.chapter_matches:
            chapter = chapter_tuple[0]
            evidence.append({
                "timestamp": getattr(chapter, "start_time", 0.0),
                "description": getattr(chapter, "summary", ""),
                "type": "chapter"
            })
        
        # Sort by timestamp
        evidence.sort(key=lambda e: e.get("timestamp", 0.0))
        
        return evidence
