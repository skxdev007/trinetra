"""VideoMAE V2 encoder for video understanding - native embeddings, no projection."""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List
from sharingan.exceptions import EncodingError


class VideoMAEEncoder:
    """Encodes frames using VideoMAE V2 in native embedding space (1024D)."""
    
    def __init__(self, model_name: str = "videomae-large", device: str = "auto"):
        """
        Initialize VideoMAE encoder.
        
        Args:
            model_name: Model variant
                - "videomae-large": VideoMAE V2-Large (~300M, 600MB, 1024D)
                - "videomae-huge": VideoMAE V2-Huge (~630M, 1.2GB, 1280D)
            device: Device to run on ("cpu", "cuda", or "auto")
        
        Raises:
            EncodingError: If model cannot be loaded
        """
        self.model_name = model_name
        self.device = self._select_device(device)
        self.model = None
        self.processor = None
        self._embedding_dim = None
        
        self._load_model()
    
    def _select_device(self, device: str) -> str:
        """Select appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self) -> None:
        """Load VideoMAE model."""
        try:
            from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
        except ImportError:
            raise EncodingError(
                "VideoMAE requires 'transformers' package. "
                "Install with: pip install transformers"
            )
        
        try:
            model_map = {
                "videomae-large": "MCG-NJU/videomae-large",
                "videomae-huge": "MCG-NJU/videomae-huge"
            }
            
            hf_model_name = model_map.get(self.model_name, "MCG-NJU/videomae-large")
            
            print(f"Loading VideoMAE model {hf_model_name}...")
            print(f"Target device: {self.device}")
            
            self.processor = VideoMAEImageProcessor.from_pretrained(hf_model_name)
            self.model = VideoMAEForVideoClassification.from_pretrained(hf_model_name)
            self.model.eval()
            self.model.to(self.device)
            
            # Get native embedding dimension from model config
            self._embedding_dim = self.model.config.hidden_size
            
            print(f"✓ VideoMAE loaded on {self.device} (native {self._embedding_dim}D embeddings)")
            
        except Exception as e:
            raise EncodingError(f"Failed to load VideoMAE: {str(e)}")
    
    @torch.no_grad()
    def encode_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Encode single frame to native embedding vector.
        
        Args:
            frame: Frame as numpy array (H, W, C) in RGB format
        
        Returns:
            Embedding vector as numpy array (native dimension, no projection)
        
        Raises:
            EncodingError: If encoding fails
        """
        try:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            pil_image = Image.fromarray(frame)
            
            # VideoMAE expects video input (list of frames)
            # We repeat the frame 16 times to create a pseudo-video
            inputs = self.processor([pil_image] * 16, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get hidden states (not classification logits)
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use last hidden state and pool over sequence dimension
            embedding = outputs.hidden_states[-1].mean(dim=1)  # Shape: (1, hidden_size)
            
            # Normalize embedding (important for cosine similarity)
            embedding = F.normalize(embedding, dim=-1)
            
            return embedding.cpu().numpy().squeeze()
            
        except Exception as e:
            raise EncodingError(f"Failed to encode frame: {str(e)}")
    
    @torch.no_grad()
    def encode_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Batch encode multiple frames.
        
        Args:
            frames: List of frames as numpy arrays
        
        Returns:
            Embeddings as numpy array of shape (N, D) where D is native dimension
        
        Raises:
            EncodingError: If encoding fails
        """
        try:
            if len(frames) == 0:
                return np.array([])
            
            # For now, encode individually (can optimize later with batching)
            embeddings = [self.encode_frame(frame) for frame in frames]
            return np.array(embeddings)
            
        except Exception as e:
            raise EncodingError(f"Failed to encode batch: {str(e)}")
    
    @property
    def embedding_dim(self) -> int:
        """Dimension of output embeddings (native, no projection)."""
        return self._embedding_dim
    
    def to(self, device: str):
        """Move model to device."""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self
    
    def __repr__(self) -> str:
        """String representation."""
        return f"VideoMAEEncoder(model={self.model_name}, device={self.device}, dim={self.embedding_dim})"
