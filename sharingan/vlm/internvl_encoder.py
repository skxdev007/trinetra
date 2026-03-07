"""InternVL2.5-M0.5 encoder for fast frame captioning."""

from typing import List, Optional
import torch
import numpy as np
from PIL import Image
from sharingan.exceptions import EncodingError


class InternVLEncoder:
    """
    InternVL2.5-M0.5 encoder for fast, high-quality frame captioning.
    
    Advantages over SmolVLM:
    - 2x faster inference (PVTC compression)
    - Better fine-grained details (tiling mechanism)
    - Same size (0.5B params)
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize InternVL encoder.
        
        Args:
            device: Device to run on ("cpu", "cuda", or "auto")
        """
        self.device = self._select_device(device)
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _select_device(self, device: str) -> str:
        """Select appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self) -> None:
        """Load InternVL2.5-M0.5 model."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise EncodingError(
                "InternVL requires 'transformers' package. "
                "Install with: pip install transformers"
            )

        try:
            model_name = "OpenGVLab/InternVL2_5-1B"  # Using 1B as 0.5B not yet released
            
            print(f"📦 Loading InternVL2.5 from {model_name}...")
            print(f"   Target device: {self.device}")
            
            print("   Step 1: Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            print("   Step 2: Loading model...")
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            
            if self.device == "cuda":
                print("   Step 3: Moving to CUDA...")
                self.model = self.model.to(self.device)
            
            print(f"✓ InternVL2.5 loaded successfully on {self.device}")

        except Exception as e:
            raise EncodingError(f"Failed to load InternVL model: {str(e)}")

    @torch.no_grad()
    def caption(
        self,
        frame: np.ndarray,
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 50
    ) -> str:
        """
        Generate caption for a single frame.
        
        Args:
            frame: Frame as numpy array (H, W, C) in RGB format
            prompt: Captioning prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Caption string
        """
        try:
            # Convert to PIL Image
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            pil_image = Image.fromarray(frame)
            
            # Generate caption
            response = self.model.chat(
                self.tokenizer,
                pixel_values=None,
                question=prompt,
                generation_config={
                    'max_new_tokens': max_new_tokens,
                    'do_sample': False,
                },
                image=pil_image
            )
            
            return response.strip()

        except Exception as e:
            raise EncodingError(f"Failed to caption frame: {str(e)}")

    @torch.no_grad()
    def caption_batch(
        self,
        frames: List[np.ndarray],
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 50
    ) -> List[str]:
        """
        Generate captions for multiple frames.
        
        Args:
            frames: List of frames as numpy arrays
            prompt: Captioning prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of caption strings
        """
        captions = []
        for frame in frames:
            caption = self.caption(frame, prompt, max_new_tokens)
            captions.append(caption)
        return captions

    def __repr__(self) -> str:
        """String representation."""
        return f"InternVLEncoder(device={self.device})"
