"""SmolVLM-500M for detailed frame descriptions."""

import torch
import numpy as np
from PIL import Image
from typing import List, Optional
from sharingan.exceptions import EncodingError


class SmolVLMEncoder:
    """
    SmolVLM-500M encoder for generating detailed frame descriptions.
    
    Unlike CLIP which only generates embeddings, SmolVLM can:
    - Generate natural language descriptions of frames
    - Answer questions about frame content
    - Provide detailed scene understanding
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize SmolVLM encoder.
        
        Args:
            device: Device to run on ("cpu", "cuda", or "auto")
        """
        self.device = self._select_device(device)
        self.model = None
        self.processor = None
        self._load_model()
    
    def _select_device(self, device: str) -> str:
        """Select appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self) -> None:
        """Load SmolVLM-500M model."""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
        except ImportError:
            raise EncodingError(
                "SmolVLM requires 'transformers' package. "
                "Install with: pip install transformers"
            )
        
        try:
            model_name = "HuggingFaceTB/SmolVLM-500M-Instruct"
            
            print(f"📦 Loading SmolVLM-500M...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            # Load model based on device
            if self.device == "cuda" and torch.cuda.is_available():
                # Use 8-bit quantization for GPU
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
                
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                print(f"✓ SmolVLM-500M loaded on {self.device} (8-bit quantized)")
            else:
                # Load without quantization for CPU
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                )
                self.model = self.model.to(self.device)
                print(f"✓ SmolVLM-500M loaded on {self.device}")
            
            self.model.eval()
            
        except Exception as e:
            raise EncodingError(f"Failed to load SmolVLM model: {str(e)}")
    
    @torch.no_grad()
    def describe_frame(
        self,
        frame: np.ndarray,
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 100
    ) -> str:
        """
        Generate detailed description of a frame.
        
        Args:
            frame: Frame as numpy array (H, W, C) in RGB format
            prompt: Question/prompt about the frame
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated description
        """
        try:
            # Convert to PIL Image
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            pil_image = Image.fromarray(frame)
            
            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            prompt_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=prompt_text,
                images=[pil_image],
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            
            # Decode
            generated_text = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Extract answer (remove prompt)
            if prompt in generated_text:
                answer = generated_text.split(prompt)[-1].strip()
            else:
                answer = generated_text.strip()
            
            return answer
            
        except Exception as e:
            raise EncodingError(f"Failed to describe frame: {str(e)}")
    
    @torch.no_grad()
    def describe_batch(
        self,
        frames: List[np.ndarray],
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 100
    ) -> List[str]:
        """
        Generate descriptions for multiple frames.
        
        Args:
            frames: List of frames as numpy arrays
            prompt: Question/prompt about the frames
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of generated descriptions
        """
        descriptions = []
        for frame in frames:
            desc = self.describe_frame(frame, prompt, max_new_tokens)
            descriptions.append(desc)
        return descriptions
    
    def __repr__(self) -> str:
        """String representation."""
        return f"SmolVLMEncoder(device={self.device})"
