"""
============================================================================
SYSTEM DESIGN: TemporalBench Dataset Loader for Evaluation
============================================================================

WHAT THIS FILE DOES:
This file loads the TemporalBench dataset, which is a benchmark for testing
how well video understanding systems can reason about time. It contains
questions like "What happened before the person picked up the cup?" or
"Why did the person leave the room?" that require understanding temporal
relationships between events in videos.

HOW IT FITS IN THE SYSTEM:
TemporalBench is used in Phase 6 (Baseline Evaluation) to compare SHARINGAN
against commercial VLMs like GPT-4o and Gemini. We process 100 temporal
reasoning questions and measure accuracy, latency, and cost. This proves
that SHARINGAN's multi-scale temporal reasoning works better than expensive
API-based models.

KEY CONCEPTS:
- TemporalBench: A dataset with videos and temporal reasoning questions
- Temporal Reasoning: Understanding "before", "after", "caused by", "why"
- Evaluation Subset: We focus on 100 questions about temporal relationships
- Ground Truth: Each question has a correct answer we compare against
- Metrics: Accuracy (% correct), latency (ms per query), cost ($ per query)

WHY IT MATTERS:
Without rigorous evaluation on TemporalBench, we can't prove SHARINGAN works
better than GPT-4o. This dataset loader enables the critical baseline
comparison that validates our entire architecture. It's the difference
between "we built something cool" and "we beat GPT-4o on temporal reasoning."

TEMPORALBENCH STRUCTURE:
- Videos: Short clips (10-60 seconds) with temporal events
- Questions: Multiple choice or open-ended temporal reasoning questions
- Question Types:
  * Gesture questions: "What gesture did the person make at 0:05?"
  * Action questions: "What action happened after the door opened?"
  * Scene questions: "What was the setting when the conversation started?"
  * Narrative questions: "Why did the person return to the earlier location?"
- Annotations: Ground truth answers and temporal bounds
- Subset: We use 100 questions focused on temporal reasoning for baseline

============================================================================
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TemporalBenchQuestion:
    """A single question from TemporalBench dataset."""
    
    question_id: str
    video_id: str
    question: str
    answer: str
    question_type: str  # "gesture", "action", "scene", "narrative"
    temporal_bounds: Optional[Tuple[float, float]]  # (start_time, end_time) in seconds
    choices: Optional[List[str]]  # For multiple choice questions
    metadata: Dict


@dataclass
class TemporalBenchSample:
    """A complete sample with video and question."""
    
    question: TemporalBenchQuestion
    video_path: str
    video_frames: Optional[np.ndarray] = None  # Lazy loaded


class TemporalBenchDataset:
    """
    Dataset loader for TemporalBench temporal reasoning benchmark.
    
    This class loads TemporalBench questions and videos for evaluation.
    It focuses on the temporal reasoning subset (100 questions) used for
    baseline comparison against GPT-4o and Gemini.
    
    Usage:
        dataset = TemporalBenchDataset(data_dir="data/temporalbench")
        
        # Load temporal reasoning subset
        questions = dataset.load_temporal_reasoning_subset()
        
        # Iterate through samples
        for sample in dataset:
            video = dataset.load_video(sample.video_path)
            prediction = model.predict(video, sample.question.question)
            accuracy = dataset.evaluate_prediction(
                sample.question.question_id,
                prediction
            )
    """
    
    def __init__(
        self,
        data_dir: str,
        subset: str = "temporal_reasoning",
        max_questions: int = 100
    ):
        """
        Initialize TemporalBench dataset loader.
        
        Args:
            data_dir: Path to TemporalBench dataset directory
            subset: Which subset to load ("temporal_reasoning", "all")
            max_questions: Maximum number of questions to load (default: 100)
        """
        self.data_dir = Path(data_dir)
        self.subset = subset
        self.max_questions = max_questions
        
        self.questions: List[TemporalBenchQuestion] = []
        self.video_dir = self.data_dir / "videos"
        self.annotations_file = self.data_dir / "annotations.json"
        
        # Load questions on initialization
        self._load_questions()
    
    def _load_questions(self) -> None:
        """Load questions from annotations file."""
        if not self.annotations_file.exists():
            raise FileNotFoundError(
                f"Annotations file not found: {self.annotations_file}\n"
                f"Please download TemporalBench dataset to {self.data_dir}"
            )
        
        with open(self.annotations_file, "r") as f:
            annotations = json.load(f)
        
        # Filter to temporal reasoning subset if requested
        if self.subset == "temporal_reasoning":
            annotations = [
                ann for ann in annotations
                if ann.get("question_type") in ["action", "scene", "narrative"]
                and self._is_temporal_reasoning_question(ann.get("question", ""))
            ]
        
        # Limit to max_questions
        annotations = annotations[:self.max_questions]
        
        # Parse annotations into TemporalBenchQuestion objects
        for ann in annotations:
            question = TemporalBenchQuestion(
                question_id=ann["question_id"],
                video_id=ann["video_id"],
                question=ann["question"],
                answer=ann["answer"],
                question_type=ann.get("question_type", "unknown"),
                temporal_bounds=self._parse_temporal_bounds(ann.get("temporal_bounds")),
                choices=ann.get("choices"),
                metadata=ann.get("metadata", {})
            )
            self.questions.append(question)
    
    def _is_temporal_reasoning_question(self, question: str) -> bool:
        """
        Check if question requires temporal reasoning.
        
        Temporal reasoning questions contain keywords like:
        - "before", "after", "during", "while"
        - "first", "then", "next", "finally"
        - "why", "caused", "because", "result"
        - "when", "what time", "how long"
        """
        temporal_keywords = [
            "before", "after", "during", "while",
            "first", "then", "next", "finally",
            "why", "caused", "because", "result",
            "when", "what time", "how long",
            "earlier", "later", "previously"
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in temporal_keywords)
    
    def _parse_temporal_bounds(
        self,
        bounds: Optional[Dict]
    ) -> Optional[Tuple[float, float]]:
        """Parse temporal bounds from annotation."""
        if bounds is None:
            return None
        
        start = bounds.get("start", 0.0)
        end = bounds.get("end", 0.0)
        
        if start == 0.0 and end == 0.0:
            return None
        
        return (float(start), float(end))
    
    def load_temporal_reasoning_subset(self) -> List[TemporalBenchQuestion]:
        """
        Load the temporal reasoning subset (100 questions).
        
        Returns:
            List of TemporalBenchQuestion objects focused on temporal reasoning
        """
        return self.questions
    
    def load_video(self, video_path: str) -> np.ndarray:
        """
        Load video frames from file.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Video frames as numpy array of shape (T, H, W, C)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames loaded from video: {video_path}")
        
        return np.array(frames)
    
    def get_video_path(self, video_id: str) -> str:
        """
        Get full path to video file.
        
        Args:
            video_id: Video identifier
        
        Returns:
            Full path to video file
        """
        # Try common video extensions
        for ext in [".mp4", ".avi", ".mov", ".webm"]:
            video_path = self.video_dir / f"{video_id}{ext}"
            if video_path.exists():
                return str(video_path)
        
        raise FileNotFoundError(
            f"Video not found for ID: {video_id}\n"
            f"Searched in: {self.video_dir}"
        )
    
    def evaluate_prediction(
        self,
        question_id: str,
        prediction: str
    ) -> Dict[str, float]:
        """
        Evaluate a model prediction against ground truth.
        
        Args:
            question_id: Question identifier
            prediction: Model's predicted answer
        
        Returns:
            Dictionary with evaluation metrics:
            - exact_match: 1.0 if exact match, 0.0 otherwise
            - contains_answer: 1.0 if prediction contains answer, 0.0 otherwise
        """
        # Find question by ID
        question = None
        for q in self.questions:
            if q.question_id == question_id:
                question = q
                break
        
        if question is None:
            raise ValueError(f"Question not found: {question_id}")
        
        # Normalize strings for comparison
        pred_normalized = prediction.strip().lower()
        answer_normalized = question.answer.strip().lower()
        
        # Exact match
        exact_match = 1.0 if pred_normalized == answer_normalized else 0.0
        
        # Contains answer (more lenient)
        contains_answer = 1.0 if answer_normalized in pred_normalized else 0.0
        
        return {
            "exact_match": exact_match,
            "contains_answer": contains_answer
        }
    
    def evaluate_batch(
        self,
        predictions: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Evaluate a batch of predictions.
        
        Args:
            predictions: Dictionary mapping question_id to predicted answer
        
        Returns:
            Dictionary with aggregate metrics:
            - accuracy_exact: Exact match accuracy
            - accuracy_contains: Contains answer accuracy
            - total_questions: Number of questions evaluated
        """
        exact_matches = []
        contains_matches = []
        
        for question_id, prediction in predictions.items():
            metrics = self.evaluate_prediction(question_id, prediction)
            exact_matches.append(metrics["exact_match"])
            contains_matches.append(metrics["contains_answer"])
        
        return {
            "accuracy_exact": np.mean(exact_matches) if exact_matches else 0.0,
            "accuracy_contains": np.mean(contains_matches) if contains_matches else 0.0,
            "total_questions": len(predictions)
        }
    
    def get_question_by_id(self, question_id: str) -> Optional[TemporalBenchQuestion]:
        """Get question by ID."""
        for question in self.questions:
            if question.question_id == question_id:
                return question
        return None
    
    def get_questions_by_type(self, question_type: str) -> List[TemporalBenchQuestion]:
        """Get all questions of a specific type."""
        return [
            q for q in self.questions
            if q.question_type == question_type
        ]
    
    def __len__(self) -> int:
        """Return number of questions in dataset."""
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> TemporalBenchSample:
        """Get sample by index."""
        question = self.questions[idx]
        video_path = self.get_video_path(question.video_id)
        
        return TemporalBenchSample(
            question=question,
            video_path=video_path
        )
    
    def __iter__(self):
        """Iterate through dataset samples."""
        for idx in range(len(self)):
            yield self[idx]
