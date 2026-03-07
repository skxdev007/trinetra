"""Lightweight LLM for conversational video queries using Qwen2.5-0.5B."""

import torch
from typing import List, Dict, Optional
import numpy as np


class VideoLLM:
    """
    Conversational interface for video understanding using Qwen2.5-0.5B.
    
    Uses retrieved video segments as context for natural language responses.
    """
    
    def __init__(self, model_name: str = "qwen-0.5b", device: str = "auto", enable_compile: bool = True):
        """
        Initialize the LLM.
        
        Args:
            model_name: Model to use ("qwen-0.5b" or "qwen-1.5b")
            device: Device to run on ("cpu", "cuda", or "auto")
            enable_compile: Enable torch.compile for 20-40% speedup (default: True)
        """
        self.model_name = model_name
        self.device = self._select_device(device)
        self.enable_compile = enable_compile
        self.model = None
        self.tokenizer = None
        self.chat_history = []
        
        self._load_model()
    
    def _select_device(self, device: str) -> str:
        """Select appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self) -> None:
        """Load Qwen2.5 model with 4-bit quantization."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            # Map model names to HuggingFace identifiers
            model_map = {
                "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
                "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct"
            }
            
            hf_model_name = model_map.get(self.model_name, "Qwen/Qwen2.5-0.5B-Instruct")
            
            print(f"📦 Loading {hf_model_name} with 4-bit quantization...")
            print(f"   Note: Downloads 3.1GB FP16 model, then quantizes to ~900MB VRAM")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                hf_model_name,
                trust_remote_code=True
            )
            
            # 4-bit quantization config (NF4 for better quality)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            # Load model with 4-bit quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            
            # Apply torch.compile for 20-40% speedup
            if self.enable_compile and hasattr(torch, 'compile'):
                print(f"🚀 Compiling model with torch.compile (mode='reduce-overhead')...")
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print(f"✓ Model compiled - expect 20-40% speedup")
                except Exception as e:
                    print(f"⚠️  Compilation failed: {e}, continuing without compilation")
            
            vram_usage = "~900MB" if self.model_name == "qwen-1.5b" else "~538MB"
            print(f"✓ {self.model_name} loaded with 4-bit quantization ({vram_usage} VRAM)")
            
        except ImportError as e:
            raise ImportError(
                "Qwen2.5 requires transformers and bitsandbytes. "
                "Install with: pip install transformers bitsandbytes accelerate"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name}: {str(e)}") from e
    
    def chat(
        self,
        query: str,
        video_context: List[Dict],
        max_new_tokens: int = 256,
        temperature: float = 0.1,  # Very low temperature for deterministic answers
        randomize_options: bool = True  # NEW: Randomize A/B to eliminate bias
    ) -> str:
        """
        Generate conversational response about video content.
        
        Args:
            query: User's question
            video_context: List of relevant video segments with timestamps and descriptions
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            randomize_options: Randomize A/B order to eliminate bias
            
        Returns:
            Generated response (with original option labels if randomized)
        """
        if self.model is None:
            return "LLM not available. Please ensure Qwen2.5 is properly installed."
        
        # Build context from video segments with action classifications
        context_text = self._build_context(video_context)
        
        # Detect if this is a multiple-choice question
        is_multiple_choice = ('A.' in query or 'A)' in query) and ('B.' in query or 'B)' in query)
        
        # Store original query for option mapping
        original_query = query
        answer_map = {'A': 'A', 'B': 'B'}
        
        if is_multiple_choice:
            # Randomize options to eliminate bias
            if randomize_options:
                import random
                if random.random() > 0.5:
                    # Swap A and B
                    query = self._swap_options(query)
                    answer_map = {'A': 'B', 'B': 'A'}
            
            # Improved prompt with EXPLICIT temporal reasoning focus
            system_prompt = (
                "You are a precise video temporal reasoning assistant. "
                "You will see a TEMPORAL SEQUENCE showing what happens in chronological order.\n\n"
                "CRITICAL: Pay attention to:\n"
                "1. TEMPORAL ORDER: What happens FIRST, THEN, FINALLY\n"
                "2. STATE CHANGES: Does light turn ON or OFF? Is screw TIGHTENED or LOOSENED?\n"
                "3. HAND USAGE: Which hand (right/left) performs each action?\n"
                "4. ACTION SEQUENCE: The ORDER of actions is critical!\n\n"
                "Compare BOTH options against the temporal sequence:\n"
                "- Does option A match the sequence shown?\n"
                "- Does option B match the sequence shown?\n"
                "- Which option has the CORRECT ORDER of events?\n\n"
                "Respond with ONLY the letter (A or B). No explanation."
            )
            
            user_prompt = f"{context_text}\n\n{query}\n\nBased on the temporal sequence above, answer: A or B?"
        else:
            # Regular conversational prompt
            system_prompt = (
                "You are a helpful video analysis assistant. "
                "Answer questions about the video based on the provided context. "
                "Be concise and specific, referencing timestamps when relevant."
            )
            
            user_prompt = f"Video Context:\n{context_text}\n\nQuestion: {query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Add chat history for context (only for non-multiple-choice)
        if self.chat_history and not is_multiple_choice:
            messages = self.chat_history[-4:] + messages  # Keep last 2 exchanges
        
        try:
            # Format with chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens if not is_multiple_choice else 10,  # Short for MC
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Update chat history (only for non-multiple-choice)
            if not is_multiple_choice:
                self.chat_history.append({"role": "user", "content": query})
                self.chat_history.append({"role": "assistant", "content": response})
            else:
                # Map answer back to original labels if we swapped
                response_clean = response.strip().upper()
                if response_clean in answer_map:
                    response = answer_map[response_clean]
            
            return response.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _build_context(self, video_context: List[Dict]) -> str:
        """
        Build rich context with EXPLICIT temporal ordering for LLM.
        
        Key improvements:
        1. Visual timeline with arrows showing sequence
        2. Explicit "HAPPENS BEFORE/AFTER" relationships
        3. Action summaries extracted from classifications
        4. Clear temporal markers (FIRST, THEN, FINALLY)
        """
        if not video_context:
            return "No relevant video segments found."
        
        # Sort by timestamp for temporal ordering
        sorted_context = sorted(video_context, key=lambda x: x.get('timestamp', 0))
        
        context_parts = []
        context_parts.append("TEMPORAL SEQUENCE (what happens in order):")
        context_parts.append("=" * 70)
        
        # Build explicit temporal sequence
        for i, segment in enumerate(sorted_context[:10], 1):  # Top 10 segments
            timestamp = segment.get('timestamp', 0)
            confidence = segment.get('confidence', 0)
            description = segment.get('description', 'Content detected')
            actions = segment.get('actions', {})
            
            mins = int(timestamp // 60)
            secs = int(timestamp % 60)
            time_str = f"{mins}:{secs:02d}"
            
            # Temporal marker
            if i == 1:
                marker = "FIRST"
            elif i == len(sorted_context[:10]):
                marker = "FINALLY"
            else:
                marker = f"THEN (step {i})"
            
            # Extract key actions
            action_summary = self._extract_action_summary(actions)
            
            # Build explicit temporal statement
            if action_summary:
                segment_text = f"{marker} at [{time_str}]: {action_summary}"
            else:
                segment_text = f"{marker} at [{time_str}]: {description}"
            
            # Add visual arrow for sequence
            if i < len(sorted_context[:10]):
                segment_text += "\n    ↓"
            
            context_parts.append(segment_text)
        
        context_parts.append("=" * 70)
        context_parts.append("\nKEY TEMPORAL RELATIONSHIPS:")
        
        # Add explicit before/after relationships for critical actions
        critical_actions = self._identify_critical_actions(sorted_context[:10])
        if critical_actions:
            for rel in critical_actions:
                context_parts.append(f"  • {rel}")
        else:
            context_parts.append("  • Sequence shown above")
        
        return "\n".join(context_parts)
    
    def _extract_action_summary(self, actions: Dict[str, str]) -> str:
        """
        Extract concise action summary from classifications.
        
        Focuses on the most discriminative actions:
        - Hand used (right/left/both)
        - Primary action (tightening/loosening/connecting/etc)
        - State changes (ON/OFF)
        """
        if not actions:
            return ""
        
        summary_parts = []
        
        # Hand used
        if 'hand_used' in actions:
            hand = actions['hand_used'].replace('a person using their ', '').replace(' hand', '')
            if hand != 'no':
                summary_parts.append(f"using {hand} hand")
        
        # Primary action
        primary_actions = ['screw_action', 'wire_action']
        for action_type in primary_actions:
            if action_type in actions:
                action = actions[action_type]
                if 'no' not in action.lower():
                    # Extract verb
                    action_clean = action.replace('a person ', '').replace('a wire', 'wire')
                    summary_parts.append(action_clean)
        
        # State changes (most important!)
        if 'light_state' in actions:
            state = actions['light_state']
            if 'turning on' in state:
                summary_parts.append("→ LIGHT TURNS ON")
            elif 'turning off' in state:
                summary_parts.append("→ LIGHT TURNS OFF")
        
        return ', '.join(summary_parts) if summary_parts else ""
    
    def _identify_critical_actions(self, segments: List[Dict]) -> List[str]:
        """
        Identify critical temporal relationships between actions.
        
        Looks for:
        - State changes (ON/OFF)
        - Action sequences (tighten THEN pull)
        - Hand switches (right THEN left)
        """
        relationships = []
        
        for i in range(len(segments) - 1):
            curr_actions = segments[i].get('actions', {})
            next_actions = segments[i+1].get('actions', {})
            
            # Check for state change
            curr_light = curr_actions.get('light_state', '')
            next_light = next_actions.get('light_state', '')
            
            if 'turning off' in curr_light and 'turning on' in next_light:
                relationships.append("Light switches from OFF to ON")
            elif 'turning on' in curr_light and 'turning off' in next_light:
                relationships.append("Light switches from ON to OFF")
            
            # Check for action sequence
            curr_screw = curr_actions.get('screw_action', '')
            next_screw = next_actions.get('screw_action', '')
            
            if 'tightening' in curr_screw and 'loosening' in next_screw:
                relationships.append("Screw is tightened BEFORE being loosened")
            elif 'loosening' in curr_screw and 'tightening' in next_screw:
                relationships.append("Screw is loosened BEFORE being tightened")
        
        return relationships[:3]  # Top 3 most important
    
    def reset_history(self):
        """Clear chat history."""
        self.chat_history = []
    
    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        """
        Generate response from a direct prompt (no chat formatting).
        
        Args:
            prompt: Direct prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        if self.model is None:
            return "LLM not available. Please ensure Qwen2.5 is properly installed."
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def __repr__(self) -> str:
        """String representation."""
        return f"VideoLLM(model={self.model_name}, device={self.device})"

    def _swap_options(self, query: str) -> str:
        """
        Swap A and B options in a multiple-choice question.
        
        Args:
            query: Original question with A and B options
            
        Returns:
            Question with swapped options
        """
        import re
        
        # Extract options A and B
        # Pattern: "A. <text>\nB. <text>" or "A) <text>\nB) <text>"
        pattern_a = r'A[\.\)]\s*(.*?)(?=\nB[\.\)])'
        pattern_b = r'B[\.\)]\s*(.*?)(?=\n|$)'
        
        match_a = re.search(pattern_a, query, re.DOTALL)
        match_b = re.search(pattern_b, query, re.DOTALL)
        
        if not match_a or not match_b:
            return query  # Can't parse, return original
        
        option_a_text = match_a.group(1).strip()
        option_b_text = match_b.group(1).strip()
        
        # Determine delimiter (. or ))
        delimiter = '.' if 'A.' in query else ')'
        
        # Swap options
        swapped = query.replace(
            f"A{delimiter} {option_a_text}",
            f"TEMP_PLACEHOLDER"
        ).replace(
            f"B{delimiter} {option_b_text}",
            f"A{delimiter} {option_b_text}"
        ).replace(
            "TEMP_PLACEHOLDER",
            f"B{delimiter} {option_a_text}"
        )
        
        return swapped
