"""Frame encoding using lightweight VLMs."""

from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sharingan.exceptions import EncodingError


class FrameEncoder:
    """Encodes frames into semantic embeddings using lightweight VLMs."""

    def __init__(self, model_name: str = "clip-vit-b32", device: str = "auto"):
        """
        Initialize frame encoder.

        Args:
            model_name: VLM model identifier
                - "clip-vit-b32": CLIP ViT-B/32 (default)
                - "clip-vit-b16": CLIP ViT-B/16
                - "clip-vit-l14": CLIP ViT-L/14
            device: Device to run on ("cpu", "cuda", or "auto")

        Raises:
            EncodingError: If model cannot be loaded
        """
        self.model_name = model_name
        self.device = self._select_device(device)
        self.model = None
        self.preprocess = None
        self._embedding_dim = None

        self._load_model()

    def _select_device(self, device: str) -> str:
        """Select appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self) -> None:
        """Load VLM model."""
        try:
            if "clip" in self.model_name.lower():
                self._load_clip_model()
            else:
                raise EncodingError(f"Unsupported model: {self.model_name}")
        except Exception as e:
            raise EncodingError(f"Failed to load model {self.model_name}: {str(e)}")

    def _load_clip_model(self) -> None:
        """Load CLIP model."""
        try:
            import clip
        except ImportError:
            raise EncodingError(
                "CLIP model requires 'clip' package. "
                "Install with: pip install git+https://github.com/openai/CLIP.git"
            )

        # Map model names to CLIP identifiers
        model_map = {
            "clip-vit-b32": "ViT-B/32",
            "clip-vit-b16": "ViT-B/16",
            "clip-vit-l14": "ViT-L/14",
        }

        clip_model_name = model_map.get(self.model_name, "ViT-B/32")

        try:
            import os
            # WSL fix: Disable JIT to avoid torch.jit.load() hanging
            os.environ['PYTORCH_JIT'] = '0'
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            
            print(f"Loading CLIP model {clip_model_name}...")
            
            # Load with jit=False to avoid WSL hanging issue
            self.model, self.preprocess = clip.load(clip_model_name, device="cpu", jit=False)
            self.model.eval()
            
            # Move to target device after loading
            if self.device != "cpu" and torch.cuda.is_available():
                print(f"Moving model to {self.device}...")
                self.model = self.model.to(self.device)
                print(f"✓ Model on {self.device}")
            else:
                print(f"✓ Model on CPU")
            
            # Get embedding dimension
            dim_map = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768}
            self._embedding_dim = dim_map.get(clip_model_name, 512)

        except Exception as e:
            raise EncodingError(f"Failed to load CLIP model: {str(e)}")

    @torch.no_grad()
    def encode_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Encode single frame to embedding vector.

        Args:
            frame: Frame as numpy array (H, W, C) in RGB format

        Returns:
            Embedding vector as numpy array

        Raises:
            EncodingError: If encoding fails
        """
        try:
            # Convert numpy array to PIL Image
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            pil_image = Image.fromarray(frame)

            # Preprocess and encode
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

            # Encode with CLIP
            if "clip" in self.model_name.lower():
                embedding = self.model.encode_image(image_tensor)
                # Normalize embedding
                embedding = F.normalize(embedding, dim=-1)
            else:
                raise EncodingError(f"Encoding not implemented for {self.model_name}")

            # Convert to numpy
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
            Embeddings as numpy array of shape (N, D)

        Raises:
            EncodingError: If encoding fails
        """
        try:
            if len(frames) == 0:
                return np.array([])

            # Convert frames to PIL Images
            pil_images = []
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(frame))

            # Preprocess batch
            image_tensors = torch.stack([
                self.preprocess(img) for img in pil_images
            ]).to(self.device)

            # Encode batch
            if "clip" in self.model_name.lower():
                embeddings = self.model.encode_image(image_tensors)
                # Normalize embeddings
                embeddings = F.normalize(embeddings, dim=-1)
            else:
                raise EncodingError(f"Batch encoding not implemented for {self.model_name}")

            # Convert to numpy
            return embeddings.cpu().numpy()

        except Exception as e:
            raise EncodingError(f"Failed to encode batch: {str(e)}")

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to embedding vector.

        Useful for text-to-video queries.

        Args:
            text: Text string to encode

        Returns:
            Embedding vector as numpy array

        Raises:
            EncodingError: If encoding fails
        """
        try:
            if "clip" in self.model_name.lower():
                import clip
                
                # Truncate text to fit CLIP's 77 token limit
                # Rough estimate: 77 tokens ≈ 300 characters
                max_chars = 300
                if len(text) > max_chars:
                    text = text[:max_chars-3] + "..."
                
                text_tokens = clip.tokenize([text], truncate=True).to(self.device)
                with torch.no_grad():
                    text_embedding = self.model.encode_text(text_tokens)
                    text_embedding = F.normalize(text_embedding, dim=-1)
                return text_embedding.cpu().numpy().squeeze()
            else:
                raise EncodingError(f"Text encoding not implemented for {self.model_name}")

        except Exception as e:
            raise EncodingError(f"Failed to encode text: {str(e)}")

    @property
    def embedding_dim(self) -> int:
        """Dimension of output embeddings."""
        return self._embedding_dim

    def to(self, device: str):
        """Move model to device."""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self

    def __repr__(self) -> str:
        """String representation."""
        return f"FrameEncoder(model={self.model_name}, device={self.device}, dim={self.embedding_dim})"
