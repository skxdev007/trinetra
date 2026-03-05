"""
VideoMAE-based processor with text-based TEG architecture.

Architecture:
    VideoMAE (1024D) → Action Classifier → Text Labels → TEG → Qwen

Key differences from CLIP processor:
- No cross-modal embedding matching
- Vision → Text translation happens once during ingest
- Query pipeline reads text, not embeddings
- Qwen operates purely on text
"""

from typing import Optional, List, Dict, Any
import numpy as np
from pathlib import Path


class VideoProcessorVideoMAE:
    """
    VideoMAE-based video processor with text-based TEG.
    
    Example:
        >>> processor = VideoProcessorVideoMAE(device='cuda')
        >>> results = processor.process('video.mp4')
        >>> response = processor.chat('What happens in this video?')
    """
    
    def __init__(
        self,
        device: str = 'auto',
        target_fps: float = 5.0,
        llm_model: str = 'qwen-1.5b',
        cache_dir: str = 'cache'
    ):
        """
        Initialize VideoMAE processor.
        
        Args:
            device: Device to use ('cpu', 'cuda', or 'auto')
            target_fps: Frames per second to process
            llm_model: Language model ('qwen-0.5b' or 'qwen-1.5b')
            cache_dir: Directory for caching
        """
        self.device = device
        self.target_fps = target_fps
        self.llm_model = llm_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # State
        self.text_teg = None  # Text-based temporal event graph
        self.timestamps = None
        self.frame_indices = None
        self.video_info = None
        
        # Models (lazy loaded)
        self._videomae = None
        self._action_classifier = None
        self._llm = None
    
    def process(self, video_path: str) -> Dict[str, Any]:
        """
        Process a video file using VideoMAE → Action Classifier → Text TEG.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with results:
                - video_info: Video metadata
                - text_teg: Text-based temporal event graph
                - timestamps: Frame timestamps
                - frame_indices: Frame indices
        """
        from sharingan.video import VideoLoader, FrameSampler
        from sharingan.vlm.videomae_encoder import VideoMAEEncoder
        from sharingan.vlm.action_classifier import ActionClassifier
        import hashlib
        import os
        
        print(f"🎬 Processing video with VideoMAE pipeline: {video_path}")
        
        # Check cache
        file_stat = os.stat(video_path)
        cache_key = f"{os.path.basename(video_path)}_{file_stat.st_size}_{int(file_stat.st_mtime)}"
        video_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_path = self.cache_dir / f"videomae_{video_hash}.json"
        
        if cache_path.exists():
            print(f"💾 Loading from cache...")
            import json
            with open(cache_path) as f:
                cached = json.load(f)
            self.text_teg = cached['text_teg']
            self.timestamps = cached['timestamps']
            self.frame_indices = cached['frame_indices']
            self.video_info = cached['video_info']
            print(f"✓ Loaded {len(self.text_teg)} text events from cache")
            return {
                'video_info': self.video_info,
                'text_teg': self.text_teg,
                'timestamps': self.timestamps,
                'frame_indices': self.frame_indices
            }
        
        # Load video
        print(f"📹 Loading video...")
        loader = VideoLoader(video_path, backend='opencv')
        sampler = FrameSampler(strategy='adaptive', target_fps=self.target_fps)
        
        # Initialize VideoMAE encoder (reuse if already loaded)
        if not self._videomae:
            print(f"🧠 Initializing VideoMAE encoder...")
            self._videomae = VideoMAEEncoder(model_name='videomae-large', device=self.device)
        
        # Initialize action classifier (reuse if already loaded)
        if not self._action_classifier:
            print(f"🎯 Initializing action classifier...")
            self._action_classifier = ActionClassifier(
                embedding_dim=self._videomae.embedding_dim,
                device=self.device
            )
        
        # Process frames
        print(f"⚙️  Processing frames...")
        frames = []
        self.timestamps = []
        self.frame_indices = []
        embeddings = []
        
        for frame_idx, frame, change_score in sampler.sample(loader, source_fps=loader.fps):
            frames.append(frame)
            self.timestamps.append(frame_idx / loader.fps)
            self.frame_indices.append(frame_idx)
            
            # Process in batches
            if len(frames) >= 32:
                batch_embs = self._videomae.encode_batch(frames)
                embeddings.extend(batch_embs)
                frames = []
        
        # Process remaining
        if frames:
            batch_embs = self._videomae.encode_batch(frames)
            embeddings.extend(batch_embs)
        
        embeddings = np.array(embeddings)
        print(f"✓ Processed {len(embeddings)} frames")
        
        # Classify actions
        print(f"🎯 Classifying actions...")
        action_predictions = self._action_classifier.classify_batch(embeddings)
        
        # Build text-based TEG
        print(f"📝 Building text-based TEG...")
        self.text_teg = []
        for i, (timestamp, frame_idx, actions) in enumerate(zip(
            self.timestamps, self.frame_indices, action_predictions
        )):
            action_label, confidence = actions[0]  # Top prediction
            
            # Create text event
            text_event = {
                'timestamp': timestamp,
                'frame_index': frame_idx,
                'action': action_label,
                'confidence': confidence,
                'text': f"[T={timestamp:.1f}s] {action_label} (conf={confidence:.2f})"
            }
            self.text_teg.append(text_event)
        
        print(f"✓ Built TEG with {len(self.text_teg)} text events")
        
        self.video_info = {
            'fps': loader.fps,
            'total_frames': loader.total_frames,
            'duration': self.timestamps[-1] if self.timestamps else 0,
            'processed_frames': len(self.frame_indices)
        }
        
        # Cache results
        print(f"💾 Caching results...")
        import json
        with open(cache_path, 'w') as f:
            json.dump({
                'text_teg': self.text_teg,
                'timestamps': self.timestamps,
                'frame_indices': self.frame_indices,
                'video_info': self.video_info
            }, f)
        print(f"✓ Cached to {cache_path}")
        
        print(f"✅ Processing complete!")
        
        return {
            'video_info': self.video_info,
            'text_teg': self.text_teg,
            'timestamps': self.timestamps,
            'frame_indices': self.frame_indices
        }
    
    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query video using text-based search (no embedding matching).
        
        Args:
            text: Query text
            top_k: Number of results to return
            
        Returns:
            List of matches with timestamps and text descriptions
        """
        if self.text_teg is None:
            raise ValueError("Process a video first using .process()")
        
        print(f"🔍 Query: '{text}'")
        
        # Simple text matching for now
        # TODO: Use semantic text similarity (sentence-transformers)
        query_lower = text.lower()
        matches = []
        
        for event in self.text_teg:
            # Simple keyword matching
            if any(word in event['action'].lower() for word in query_lower.split()):
                matches.append({
                    'timestamp': event['timestamp'],
                    'frame': event['frame_index'],
                    'action': event['action'],
                    'confidence': event['confidence'],
                    'text': event['text']
                })
        
        # Return top-k by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        results = matches[:top_k]
        
        print(f"✓ Found {len(results)} results")
        return results
    
    def chat(self, question: str) -> str:
        """
        Chat about the video using Qwen (text-only, no embeddings).
        
        Args:
            question: Question about the video
            
        Returns:
            Response text
        """
        if self.text_teg is None:
            raise ValueError("Process a video first using .process()")
        
        # Get relevant text events
        relevant_events = self.query(question, top_k=10)
        
        if not relevant_events:
            return "No relevant events found in the video."
        
        # Build text context for LLM
        context_lines = []
        for event in relevant_events:
            context_lines.append(event['text'])
        
        context = "\n".join(context_lines)
        
        # Use LLM
        from sharingan.chat import VideoLLM
        
        if not self._llm:
            print(f"🤖 Initializing {self.llm_model}...")
            self._llm = VideoLLM(model_name=self.llm_model, device=self.device)
        
        # Create prompt
        prompt = f"""Based on the following video events, answer the question.

Video Events:
{context}

Question: {question}

Answer:"""
        
        # Generate response
        response = self._llm.generate(prompt, max_new_tokens=50, temperature=0.3)  # Reduced tokens, lower temp for faster generation
        return response
    
    def reset_chat(self):
        """Reset chat history."""
        if self._llm:
            self._llm.reset_history()
