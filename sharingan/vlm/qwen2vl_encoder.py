"""
Qwen2-VL Vision Encoder for SHARINGAN.

Extracts ONLY the vision encoder from Qwen2-VL, discarding the LLM part.
This allows us to use Qwen2-VL's superior visual understanding while keeping
our own temporal reasoning (TAS, GRU) and small LLM (Qwen-1.5B).

Architecture:
    Frame → Qwen2-VL Vision Encoder → 1024D embedding
                                            ↓
                                    Your TAS + GRU pipeline
                                            ↓
                                    Your Qwen-1.5B LLM

Benefits:
- Better visual features than CLIP (fine-grained understanding)
- Keep your temporal architecture (TAS, GRU, magnet suppression)
- Keep your fast query system
- Only ~2GB VRAM for vision encoder (vs 8GB+ for full Qwen2-VL)
"""

import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np
from PIL import Image


class Qwen2VLVisionEncoder:
    """
    Vision-only encoder using Qwen2-VL's vision tower.
    
    Extracts the vision encoder and discards the LLM, giving us
    high-quality visual embeddings while keeping memory usage low.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "auto",
        output_dim: int = 512  # Project to 512D to match your pipeline
    ):
        """
        Initialize Qwen2-VL vision encoder.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
            output_dim: Output embedding dimension (512 to match CLIP)
        """
        self.model_name = model_name
        self.device = self._select_device(device)
        self.output_dim = output_dim
        
        self.vision_model = None
        self.processor = None
        self.projection = None  # Project from Qwen2-VL dim to 512D
        
        self._load_model()
    
    def _select_device(self, device: str) -> str:
        """Select appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self) -> None:
        """Load only the vision encoder from Qwen2-VL."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            print(f"📦 Loading Qwen2-VL vision encoder from {self.model_name}...")
            print(f"   Note: Loading full model first, then extracting vision tower")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load full model (we'll extract vision tower)
            full_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            
            # Extract vision tower
            self.vision_model = full_model.visual
            self.vision_model.eval()
            
            # Get vision embedding dimension
            # Qwen2-VL-2B vision tower outputs 1536D embeddings
            vision_dim = 1536  # Qwen2-VL-2B
            
            # Create projection layer to match your pipeline (512D)
            self.projection = nn.Linear(vision_dim, self.output_dim).to(self.device)
            
            # Initialize projection with Xavier
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
            
            print(f"✓ Qwen2-VL vision encoder loaded")
            print(f"   Vision dim: {vision_dim}D → Projected to {self.output_dim}D")
            print(f"   Memory: ~2GB VRAM (vision tower only)")
            
            # Delete full model to save memory (keep only vision tower)
            del full_model
            torch.cuda.empty_cache()
            
        except ImportError as e:
            raise ImportError(
                "Qwen2-VL requires transformers>=4.37.0 and qwen-vl-utils. "
                "Install with: pip install transformers>=4.37.0 qwen-vl-utils"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen2-VL vision encoder: {str(e)}") from e
    
    def encode_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Encode a batch of frames.
        
        Args:
            frames: List of frames as numpy arrays (H, W, 3)
            
        Returns:
            Embeddings as numpy array (N, 512)
        """
        if self.vision_model is None:
            raise RuntimeError("Vision model not loaded")
        
        # Convert numpy arrays to PIL Images
        pil_images = [Image.fromarray(frame) for frame in frames]
        
        # Process images
        # Qwen2-VL processor expects messages format, but we only need vision
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"
        ).to(self.device)
        
        # Extract vision features
        with torch.no_grad():
            # Get vision embeddings
            vision_outputs = self.vision_model(
                pixel_values=inputs['pixel_values'],
                image_grid_thw=inputs.get('image_grid_thw')
            )
            
            # vision_outputs is (batch, seq_len, vision_dim)
            # Take mean over sequence dimension to get frame-level embedding
            embeddings = vision_outputs.mean(dim=1)  # (batch, vision_dim)
            
            # Project to 512D
            embeddings = self.projection(embeddings)  # (batch, 512)
            
            # L2 normalize (to match CLIP behavior)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy()
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text query.
        
        For Qwen2-VL vision-only mode, we use a simple text encoder.
        Alternatively, you can keep using CLIP's text encoder.
        
        Args:
            text: Query text
            
        Returns:
            Text embedding (512,)
        """
        # Option 1: Use CLIP text encoder (recommended for compatibility)
        from .encoder import FrameEncoder
        clip_encoder = FrameEncoder(model_name='clip-vit-b32', device=self.device)
        return clip_encoder.encode_text(text)
        
        # Option 2: Use Qwen2-VL's text encoder (more complex, not implemented here)
        # Would require loading the full model's text tower
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Qwen2VLVisionEncoder(\n"
            f"  model={self.model_name},\n"
            f"  device={self.device},\n"
            f"  output_dim={self.output_dim}\n"
            f")"
        )


class InternVL2VisionEncoder:
    """
    Vision-only encoder using InternVL2's vision tower.
    
    Similar to Qwen2-VL but uses InternVL2's InternViT-300M vision encoder.
    """
    
    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL2-2B",
        device: str = "auto",
        output_dim: int = 512
    ):
        """
        Initialize InternVL2 vision encoder.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
            output_dim: Output embedding dimension
        """
        self.model_name = model_name
        self.device = self._select_device(device)
        self.output_dim = output_dim
        
        self.vision_model = None
        self.processor = None
        self.projection = None
        
        self._load_model()
    
    def _select_device(self, device: str) -> str:
        """Select appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self) -> None:
        """Load only the vision encoder from InternVL2."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            print(f"📦 Loading InternVL2 vision encoder from {self.model_name}...")
            
            # Load full model
            full_model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            
            # Extract vision tower (InternViT)
            self.vision_model = full_model.vision_model
            self.vision_model.eval()
            
            # InternVL2-2B uses InternViT-300M which outputs 1024D
            vision_dim = 1024
            
            # Create projection
            self.projection = nn.Linear(vision_dim, self.output_dim).to(self.device)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
            
            print(f"✓ InternVL2 vision encoder loaded")
            print(f"   Vision dim: {vision_dim}D → Projected to {self.output_dim}D")
            print(f"   Memory: ~1.5GB VRAM (InternViT-300M only)")
            
            # Delete full model
            del full_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load InternVL2 vision encoder: {str(e)}") from e
    
    def encode_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        """Encode frames using InternVL2 vision tower."""
        if self.vision_model is None:
            raise RuntimeError("Vision model not loaded")
        
        # Convert to PIL and preprocess
        from torchvision import transforms
        
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((448, 448)),  # InternVL2 uses 448x448
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Process frames
        tensors = torch.stack([preprocess(frame) for frame in frames]).to(self.device)
        
        with torch.no_grad():
            # Get vision features
            vision_outputs = self.vision_model(tensors)
            
            # Take CLS token or mean pooling
            if hasattr(vision_outputs, 'last_hidden_state'):
                embeddings = vision_outputs.last_hidden_state[:, 0]  # CLS token
            else:
                embeddings = vision_outputs.mean(dim=1)
            
            # Project and normalize
            embeddings = self.projection(embeddings)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy()
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using CLIP (for compatibility)."""
        from .encoder import FrameEncoder
        clip_encoder = FrameEncoder(model_name='clip-vit-b32', device=self.device)
        return clip_encoder.encode_text(text)
    
    def __repr__(self) -> str:
        return f"InternVL2VisionEncoder(model={self.model_name}, device={self.device})"


def test_qwen2vl_vision_encoder():
    """Test Qwen2-VL vision encoder."""
    print("\n" + "="*80)
    print("Testing Qwen2-VL Vision Encoder")
    print("="*80)
    
    # Create encoder
    encoder = Qwen2VLVisionEncoder(device='auto')
    
    # Create dummy frames
    frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(4)]
    
    # Encode
    print(f"\nEncoding {len(frames)} frames...")
    embeddings = encoder.encode_batch(frames)
    
    print(f"✓ Embeddings shape: {embeddings.shape}")
    print(f"✓ Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    print(f"✓ Embedding norm: {np.linalg.norm(embeddings[0]):.3f}")
    
    # Test text encoding
    text_emb = encoder.encode_text("a person holding a screwdriver")
    print(f"✓ Text embedding shape: {text_emb.shape}")
    
    # Test similarity
    similarity = np.dot(embeddings[0], text_emb)
    print(f"✓ Frame-text similarity: {similarity:.3f}")
    
    print("\n" + "="*80)
    print("Qwen2-VL Vision Encoder Test Complete!")
    print("="*80)


if __name__ == "__main__":
    test_qwen2vl_vision_encoder()
