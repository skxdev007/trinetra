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
                - "siglip-base": SigLIP Base (768D)
                - "siglip-large": SigLIP Large (1152D)
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
            elif "siglip" in self.model_name.lower():
                self._load_siglip_model()
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
            print(f"Target device: {self.device}")
            
            # Load with jit=False to avoid WSL hanging issue
            print(f"Step 1: Loading model to CPU first...")
            self.model, self.preprocess = clip.load(clip_model_name, device="cpu", jit=False)
            print(f"Step 2: Setting model to eval mode...")
            self.model.eval()
            
            # Move to target device after loading
            if self.device != "cpu" and torch.cuda.is_available():
                print(f"Step 3: Moving model to {self.device}...")
                self.model = self.model.to(self.device)
                print(f"[OK] Model successfully loaded on {self.device}")
            else:
                print(f"[OK] Model successfully loaded on CPU")
            
            # Get embedding dimension
            dim_map = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768}
            self._embedding_dim = dim_map.get(clip_model_name, 512)

        except Exception as e:
            raise EncodingError(f"Failed to load CLIP model: {str(e)}")

    def _load_siglip_model(self) -> None:
        """Load SigLIP model."""
        try:
            from transformers import AutoProcessor, AutoModel
        except ImportError:
            raise EncodingError(
                "SigLIP model requires 'transformers' package. "
                "Install with: pip install transformers"
            )

        # Map model names to HuggingFace identifiers
        model_map = {
            "siglip-base": "google/siglip-base-patch16-224",
            "siglip-large": "google/siglip-large-patch16-256",
        }

        hf_model_name = model_map.get(self.model_name, "google/siglip-base-patch16-224")

        try:
            print(f"Loading SigLIP model {hf_model_name}...")
            print(f"Target device: {self.device}")
            
            print(f"Step 1: Loading processor...")
            self.preprocess = AutoProcessor.from_pretrained(hf_model_name)
            
            print(f"Step 2: Loading model...")
            self.model = AutoModel.from_pretrained(hf_model_name)
            self.model.eval()
            
            # Move to target device
            if self.device != "cpu" and torch.cuda.is_available():
                print(f"Step 3: Moving model to {self.device}...")
                self.model = self.model.to(self.device)
                print(f"[OK] Model successfully loaded on {self.device}")
            else:
                print(f"[OK] Model successfully loaded on CPU")
            
            # Get embedding dimension
            dim_map = {
                "google/siglip-base-patch16-224": 768,
                "google/siglip-large-patch16-256": 1152,
            }
            self._embedding_dim = dim_map.get(hf_model_name, 768)

        except Exception as e:
            raise EncodingError(f"Failed to load SigLIP model: {str(e)}")

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
            if "clip" in self.model_name.lower():
                image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            elif "siglip" in self.model_name.lower():
                inputs = self.preprocess(images=pil_image, return_tensors="pt")
                image_tensor = inputs["pixel_values"].to(self.device)
            else:
                raise EncodingError(f"Preprocessing not implemented for {self.model_name}")

            # Encode with CLIP or SigLIP
            if "clip" in self.model_name.lower():
                embedding = self.model.encode_image(image_tensor)
                # Normalize embedding
                embedding = F.normalize(embedding, dim=-1)
            elif "siglip" in self.model_name.lower():
                outputs = self.model.get_image_features(pixel_values=image_tensor)
                # Normalize embedding
                embedding = F.normalize(outputs, dim=-1)
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
            if "clip" in self.model_name.lower():
                image_tensors = torch.stack([
                    self.preprocess(img) for img in pil_images
                ]).to(self.device)
            elif "siglip" in self.model_name.lower():
                inputs = self.preprocess(images=pil_images, return_tensors="pt")
                image_tensors = inputs["pixel_values"].to(self.device)
            else:
                raise EncodingError(f"Batch preprocessing not implemented for {self.model_name}")

            # Encode batch
            if "clip" in self.model_name.lower():
                embeddings = self.model.encode_image(image_tensors)
                # Normalize embeddings
                embeddings = F.normalize(embeddings, dim=-1)
            elif "siglip" in self.model_name.lower():
                embeddings = self.model.get_image_features(pixel_values=image_tensors)
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
            elif "siglip" in self.model_name.lower():
                # Truncate text to fit SigLIP's token limit
                max_chars = 300
                if len(text) > max_chars:
                    text = text[:max_chars-3] + "..."
                
                inputs = self.preprocess(text=[text], return_tensors="pt", padding="max_length", truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    text_embedding = self.model.get_text_features(**inputs)
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
