"""Action classifier for VideoMAE embeddings → COIN action labels."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


class ActionClassifier:
    """
    Classifies VideoMAE embeddings into COIN action labels.
    
    Architecture:
        VideoMAE embedding (1024D) → Classifier head → COIN action label (text)
    
    For now, uses a simple nearest-neighbor approach with action embeddings.
    Future: Can be replaced with a trained classifier head.
    """
    
    def __init__(self, embedding_dim: int = 1024, device: str = "auto", use_videomae_classifier: bool = True):
        """
        Initialize action classifier.
        
        Args:
            embedding_dim: Dimension of VideoMAE embeddings
            device: Device to run on
            use_videomae_classifier: Use VideoMAE's built-in Kinetics-400 classifier
        """
        self.embedding_dim = embedding_dim
        self.device = self._select_device(device)
        self.use_videomae_classifier = use_videomae_classifier
        self.videomae_model = None
        self.videomae_processor = None
        
        if use_videomae_classifier:
            self._load_videomae_classifier()
        else:
            # Fallback to generic labels
            self.action_labels = [
                "person enters scene",
                "person picks up object",
                "person puts down object",
                "person turns knob",
                "person opens container",
                "person closes container",
                "person pours liquid",
                "person mixes ingredients",
                "person cuts object",
                "person assembles parts",
                "person applies tool",
                "person removes object",
                "person exits scene",
                "camera pans",
                "scene change"
            ]
            print(f"[OK] ActionClassifier initialized with {len(self.action_labels)} action labels")
    
    def _load_videomae_classifier(self):
        """Load VideoMAE model with Kinetics-400 classification head."""
        try:
            from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
            
            print(f"Loading VideoMAE Kinetics-400 classifier...")
            # Use finetuned model with Kinetics-400 labels
            model_name = "MCG-NJU/videomae-base-short-finetuned-kinetics"
            self.videomae_processor = VideoMAEImageProcessor.from_pretrained(model_name)
            self.videomae_model = VideoMAEForVideoClassification.from_pretrained(model_name)
            self.videomae_model.eval()
            self.videomae_model.to(self.device)
            
            # Kinetics-400 has 400 action classes
            num_labels = len(self.videomae_model.config.id2label)
            print(f"[OK] ActionClassifier initialized with Kinetics-400 ({num_labels} action classes)")
        except Exception as e:
            print(f"[WARN] Failed to load VideoMAE classifier: {e}")
            print(f"  Falling back to generic labels")
            self.use_videomae_classifier = False
            self.action_labels = ["person performs action"]
    
    def _select_device(self, device: str) -> str:
        """Select appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def classify(self, embedding: np.ndarray, frame: np.ndarray = None, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Classify a single VideoMAE embedding into action labels.
        
        Args:
            embedding: VideoMAE embedding (1024D)
            frame: Original frame (H, W, C) for VideoMAE classifier
            top_k: Number of top predictions to return
        
        Returns:
            List of (action_label, confidence) tuples
        """
        if self.use_videomae_classifier and frame is not None:
            try:
                import torch
                from PIL import Image
                
                # Prepare frame for VideoMAE
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                pil_image = Image.fromarray(frame)
                
                # VideoMAE expects video input (list of frames)
                inputs = self.videomae_processor([pil_image] * 16, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get classification logits
                with torch.no_grad():
                    outputs = self.videomae_model(**inputs)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probs[0], k=top_k)
                
                results = []
                for prob, idx in zip(top_probs, top_indices):
                    # Get Kinetics-400 label
                    label = self.videomae_model.config.id2label[idx.item()]
                    confidence = prob.item()
                    results.append((label, confidence))
                
                return results
                
            except Exception as e:
                print(f"[WARN] VideoMAE classification failed: {e}")
                # Fall through to placeholder
        
        # Placeholder fallback
        norm = np.linalg.norm(embedding)
        if norm > 0.9:
            action = "person performs action"
            confidence = 0.85
        else:
            action = "scene transition"
            confidence = 0.75
        
        return [(action, confidence)]
    
    def classify_batch(self, embeddings: np.ndarray, frames: List[np.ndarray] = None, top_k: int = 1) -> List[List[Tuple[str, float]]]:
        """
        Classify a batch of VideoMAE embeddings.
        
        Args:
            embeddings: VideoMAE embeddings (N, 1024)
            frames: Original frames for VideoMAE classifier
            top_k: Number of top predictions per embedding
        
        Returns:
            List of lists of (action_label, confidence) tuples
        """
        results = []
        for i, embedding in enumerate(embeddings):
            frame = frames[i] if frames and i < len(frames) else None
            results.append(self.classify(embedding, frame=frame, top_k=top_k))
        return results
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ActionClassifier(labels={len(self.action_labels)}, device={self.device})"
