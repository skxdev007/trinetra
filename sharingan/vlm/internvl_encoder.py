"""InternVL2.5-M0.5 encoder for fast frame captioning."""

from typing import List, Optional
import torch
import numpy as np
from PIL import Image
from sharingan.exceptions import EncodingError


class InternVLEncoder:
    """
    InternVL2.5 encoder for fast, high-quality frame captioning.
    
    Supports multiple model sizes with 4-bit quantization:
    - 1B: Fast, good for general captions (~2GB VRAM)
    - 4B: Better fine-grained perception (~2.5GB VRAM with 4-bit)
    - 8B: Best quality (~4.5GB VRAM with 4-bit)
    
    Advantages over SmolVLM:
    - 2x faster inference (PVTC compression)
    - Better fine-grained details (tiling mechanism)
    - 4-bit quantization for larger models
    """

    def __init__(
        self, 
        device: str = "auto", 
        enable_compile: bool = True,
        model_size: str = "4b",
        use_4bit: bool = True,
        caption_prompt: str = None
    ):
        """
        Initialize InternVL encoder.
        
        Args:
            device: Device to run on ("cpu", "cuda", or "auto")
            enable_compile: Enable torch.compile for 20-40% speedup
            model_size: Model size ("1b", "4b", "8b")
            use_4bit: Use 4-bit quantization (recommended for 4b/8b, default: True)
            caption_prompt: Custom prompt template for captioning (default: TemporalBench optimized)
        """
        self.device = self._select_device(device)
        self.enable_compile = enable_compile
        self.model_size = model_size
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        
        # Default prompt optimized for TemporalBench (ORDER, DIRECTION, STATE, HAND, COUNT)
        self.default_caption_prompt = """Analyze this frame and answer EXACTLY:

1. HAND: Which hand holds the tool? (left/right/both/neither)
2. TOOL: What tool is visible? (screwdriver/wrench/knife/none)
3. ACTION: What is happening? (tightening/loosening/pushing/pulling/connecting/disconnecting)
4. DIRECTION: Which direction? (clockwise/counterclockwise/left-to-right/right-to-left/toward/away)
5. STATE: What is the current state?
   - Light: ON/OFF/not visible
   - Screw: tight/loose/not visible
   - Wire: connected/disconnected/not visible
6. COUNT: How many times has this action occurred? (first time/second time/third time)
7. EVENT: What JUST changed in this moment? (light turned ON, wire pushed onto connector, screw became tight)

Format: "HAND: right | TOOL: screwdriver | ACTION: tightening | DIRECTION: clockwise | STATE: light=ON, screw=tight | COUNT: second time | EVENT: screw became tight"

Be EXACT. Use structured format. Max 80 words."""
        
        # Allow user to override with custom prompt
        self.caption_prompt = caption_prompt if caption_prompt else self.default_caption_prompt
        
        self._load_model()

    def _select_device(self, device: str) -> str:
        """Select appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self) -> None:
        """Load InternVL2.5 model with optional 4-bit quantization."""
        try:
            from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
        except ImportError:
            raise EncodingError(
                "InternVL requires 'transformers' and 'torchvision' packages. "
                "Install with: pip install transformers torchvision"
            )

        try:
            # Map model size to HuggingFace model name
            model_map = {
                "1b": "OpenGVLab/InternVL2_5-1B",
                "2b": "OpenGVLab/InternVL2-2B",  # InternVL 2.0 (smaller download)
                "2b-v2.5": "OpenGVLab/InternVL2_5-2B",  # InternVL 2.5
                "4b": "OpenGVLab/InternVL2_5-4B",
                "8b": "OpenGVLab/InternVL2_5-8B"
            }
            
            model_name = model_map.get(self.model_size, "OpenGVLab/InternVL2_5-1B")
            
            print(f"📦 Loading InternVL2.5-{self.model_size.upper()} from {model_name}...")
            if self.use_4bit:
                print(f"   Using 4-bit quantization (NF4)")
            print(f"   Target device: {self.device}")
            
            print("   Step 1: Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            print("   Step 2: Loading model...")
            
            if self.use_4bit and self.device == "cuda":
                # 4-bit quantization config (NF4 for better quality)
                # Use float16 for compute dtype to avoid BFloat16/Half mismatch
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,  # Use float16 consistently
                    bnb_4bit_use_double_quant=True,
                )
                
                self.model = AutoModel.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,  # Ensure consistent dtype
                    attn_implementation="eager"  # Avoid flash attention dtype issues
                )
                
                vram_estimate = {
                    "1b": "~1.2GB",
                    "2b": "~1.5GB",
                    "2b-v2.5": "~1.5GB",
                    "4b": "~2.5GB", 
                    "8b": "~4.5GB"
                }.get(self.model_size, "~2GB")
                
                print(f"   ✓ Loaded with 4-bit quantization ({vram_estimate} VRAM)")
            else:
                # Load without quantization
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager"  # Avoid flash attention issues
                )
                
                if self.device == "cuda":
                    print("   Step 3: Moving to CUDA...")
                    self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Apply torch.compile for 20-40% speedup
            if self.enable_compile and hasattr(torch, 'compile'):
                print("   Step 4: Compiling with torch.compile...")
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("   ✓ Model compiled for faster inference")
                except Exception as e:
                    print(f"   ⚠️  Compilation failed: {e}")
            
            # Build transform for image preprocessing
            IMAGENET_MEAN = (0.485, 0.456, 0.406)
            IMAGENET_STD = (0.229, 0.224, 0.225)
            
            self.build_transform = lambda input_size: T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            
            print(f"✓ InternVL2.5-{self.model_size.upper()} loaded successfully on {self.device}")

        except Exception as e:
            raise EncodingError(f"Failed to load InternVL model: {str(e)}")

    @torch.no_grad()
    def caption(
        self,
        frame: np.ndarray,
        prompt: str = None,
        max_new_tokens: int = 100
    ) -> str:
        """
        Generate caption for a single frame.
        
        Args:
            frame: Frame as numpy array (H, W, C) in RGB format
            prompt: Captioning prompt (default: uses self.caption_prompt)
            max_new_tokens: Maximum tokens to generate (default: 100 for structured output)
            
        Returns:
            Caption string
        """
        # Use default prompt if none provided
        if prompt is None:
            prompt = self.caption_prompt
        try:
            # Convert to PIL Image
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            pil_image = Image.fromarray(frame)
            
            # Preprocess image
            pixel_values = self.build_transform(448)(pil_image).unsqueeze(0)
            if self.device == "cuda":
                # Match model dtype: bfloat16 for non-quantized, float16 for 4-bit
                if self.use_4bit:
                    pixel_values = pixel_values.to(torch.float16).cuda()
                else:
                    pixel_values = pixel_values.to(torch.bfloat16).cuda()
            
            # Generate caption using InternVL's chat interface
            generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
            
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config
            )
            
            return response.strip()

        except Exception as e:
            raise EncodingError(f"Failed to caption frame: {str(e)}")

    @torch.no_grad()
    def caption_batch(
        self,
        frames: List[np.ndarray],
        prompt: str = None,
        max_new_tokens: int = 100
    ) -> List[str]:
        """
        Generate captions for multiple frames.
        
        Args:
            frames: List of frames as numpy arrays
            prompt: Captioning prompt (default: uses self.caption_prompt)
            max_new_tokens: Maximum tokens to generate (default: 100 for structured output)
            
        Returns:
            List of caption strings
        """
        captions = []
        for frame in frames:
            caption = self.caption(frame, prompt, max_new_tokens)
            captions.append(caption)
        return captions
    
    def set_caption_prompt(self, prompt: str):
        """
        Update the caption prompt template.
        
        Args:
            prompt: New prompt template
        """
        self.caption_prompt = prompt
        print(f"✓ Caption prompt updated")
    
    def reset_caption_prompt(self):
        """Reset to default TemporalBench-optimized prompt."""
        self.caption_prompt = self.default_caption_prompt
        print(f"✓ Caption prompt reset to default (TemporalBench optimized)")

    def __repr__(self) -> str:
        """String representation."""
        return f"InternVLEncoder(device={self.device})"
