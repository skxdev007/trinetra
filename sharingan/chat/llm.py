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
            # ALWAYS randomize options to eliminate bias
            if randomize_options:
                import random
                if random.random() > 0.5:
                    # Swap A and B
                    query = self._swap_options(query)
                    answer_map = {'A': 'B', 'B': 'A'}
            
            # IMPROVED prompt with explicit temporal reasoning steps
            system_prompt = (
                "You are a precise video temporal reasoning expert. "
                "Your task is to determine the CORRECT ORDER of events.\n\n"
                "TEMPORAL REASONING PROTOCOL:\n"
                "1. READ the EVENT SEQUENCE - events are listed in CHRONOLOGICAL ORDER (earliest to latest)\n"
                "2. IDENTIFY the timestamps - earlier timestamp = happened FIRST\n"
                "3. EXTRACT key attributes from each event:\n"
                "   - HAND: Which hand? (left/right/both)\n"
                "   - ACTION: What action? (tightening/loosening/pushing/pulling)\n"
                "   - DIRECTION: Which way? (clockwise/counterclockwise/onto/off)\n"
                "   - STATE: What state? (light ON/OFF, screw tight/loose, wire connected/disconnected)\n"
                "   - COUNT: How many times? (first/second/third)\n"
                "4. COMPARE both options against the timeline:\n"
                "   - For 'THEN' questions: Check if Event A timestamp < Event B timestamp\n"
                "   - For 'BEFORE/AFTER' questions: Compare timestamps directly\n"
                "   - For 'FIRST/LAST' questions: Use Event 1 (earliest) or Event N (latest)\n"
                "5. MATCH the sequence:\n"
                "   - Does option A match the chronological order?\n"
                "   - Does option B match the chronological order?\n"
                "6. The ONLY difference is usually ORDER, DIRECTION, STATE, or HAND\n"
                "7. Pay attention to: FIRST, THEN, FINALLY, BEFORE, AFTER markers\n\n"
                "RESPOND WITH ONLY THE LETTER (A or B). NO EXPLANATION."
            )
            
            user_prompt = f"{context_text}\n\n{query}\n\nBased on the timeline above, which option matches the sequence? Answer: A or B?"
            
            # DEBUG: Print what's being sent to LLM
            print(f"\n{'='*80}")
            print(f"DEBUG: LLM INPUT")
            print(f"{'='*80}")
            print(f"SYSTEM PROMPT:\n{system_prompt}\n")
            print(f"{'='*80}")
            print(f"CONTEXT:\n{context_text}\n")
            print(f"{'='*80}")
            print(f"QUESTION:\n{query}\n")
            print(f"{'='*80}\n")
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
        Build EVENT-GROUPED context for temporal ordering questions.
        
        Key improvements over frame-level:
        1. Group consecutive similar frames into events (reduces redundancy)
        2. Emphasize action boundaries (when things START and STOP)
        3. Show event duration (helps distinguish brief vs sustained actions)
        4. Clear transitions between different actions
        
        Conservative grouping: Only group frames with IDENTICAL actions
        to avoid losing important temporal details.
        """
        if not video_context:
            return "No relevant video segments found."
        
        # Sort by timestamp for temporal ordering
        sorted_context = sorted(video_context, key=lambda x: x.get('timestamp', 0))
        
        # Group frames into events
        events = self._group_frames_into_events(sorted_context[:10])
        
        context_parts = []
        context_parts.append("📹 VIDEO SEQUENCE (event-based):")
        context_parts.append("=" * 70)
        
        # Build EVENT-by-EVENT sequence
        for i, event in enumerate(events, 1):
            start_time = event['start_timestamp']
            end_time = event['end_timestamp']
            duration = end_time - start_time
            action = event['action']
            frame_count = event['frame_count']
            
            # Format timestamps
            start_mins = int(start_time // 60)
            start_secs = int(start_time % 60)
            end_mins = int(end_time // 60)
            end_secs = int(end_time % 60)
            
            # Build event text
            if frame_count == 1:
                # Single frame = brief action
                event_text = f"EVENT {i} [{start_mins}:{start_secs:02d}]: {action}"
            else:
                # Multiple frames = sustained action
                event_text = f"EVENT {i} [{start_mins}:{start_secs:02d}-{end_mins}:{end_secs:02d}]: {action} ({duration:.1f}s, {frame_count} frames)"
            
            # Add state change marker if present
            if event.get('state_change'):
                event_text += f" ⚡ {event['state_change']}"
            
            context_parts.append(event_text)
            
            # Add arrow between events
            if i < len(events):
                context_parts.append("         ↓")
        
        context_parts.append("=" * 70)
        
        # Add critical sequence summary
        context_parts.append("\n🎯 ACTION SEQUENCE:")
        action_sequence = " → ".join([e['action_short'] for e in events])
        context_parts.append(f"  {action_sequence}")
        
        # Add timing summary
        context_parts.append("\n⏱️ TIMING:")
        for i, event in enumerate(events, 1):
            duration = event['end_timestamp'] - event['start_timestamp']
            context_parts.append(f"  • Event {i}: {duration:.1f}s ({event['frame_count']} frames)")
        
        # Add state transitions
        context_parts.append("\n⚡ STATE TRANSITIONS:")
        transitions = [e['state_change'] for e in events if e.get('state_change')]
        if transitions:
            for trans in transitions:
                context_parts.append(f"  • {trans}")
        else:
            context_parts.append("  • No major state changes detected")
        
        return "\n".join(context_parts)
    
    def _group_frames_into_events(self, segments: List[Dict]) -> List[Dict]:
        """
        Group consecutive frames with similar actions into events.
        
        Conservative grouping strategy:
        - Only group frames with VERY similar actions (>80% word overlap)
        - Preserve temporal boundaries (don't merge distant frames)
        - Keep state changes as separate events
        
        Returns list of events with:
        - start_timestamp, end_timestamp
        - action (full description)
        - action_short (verb only, for summary)
        - frame_count
        - state_change (if any)
        """
        if not segments:
            return []
        
        events = []
        current_event = None
        
        for seg in segments:
            timestamp = seg.get('timestamp', 0)
            description = seg.get('description', 'Content detected')
            action = self._extract_key_action(description)
            action_short = self._extract_action_short(description)
            state_change = ""
            
            if current_event is None:
                # Start first event
                current_event = {
                    'start_timestamp': timestamp,
                    'end_timestamp': timestamp,
                    'action': action,
                    'action_short': action_short,
                    'frame_count': 1,
                    'frames': [seg],
                    'state_change': ""
                }
            else:
                # Check if this frame continues the current event
                prev_action_short = current_event['action_short']
                time_gap = timestamp - current_event['end_timestamp']
                
                # Group if: same action AND close in time (< 10s gap)
                if self._is_same_action(prev_action_short, action_short) and time_gap < 10:
                    # Continue current event
                    current_event['end_timestamp'] = timestamp
                    current_event['frame_count'] += 1
                    current_event['frames'].append(seg)
                else:
                    # Action changed or time gap too large
                    # Detect state change between events
                    if current_event['frames'] and seg:
                        prev_desc = current_event['frames'][-1].get('description', '')
                        curr_desc = description
                        state_change = self._detect_state_change(prev_desc, curr_desc)
                    
                    # Save current event
                    events.append(current_event)
                    
                    # Start new event
                    current_event = {
                        'start_timestamp': timestamp,
                        'end_timestamp': timestamp,
                        'action': action,
                        'action_short': action_short,
                        'frame_count': 1,
                        'frames': [seg],
                        'state_change': state_change
                    }
        
        # Don't forget the last event
        if current_event:
            events.append(current_event)
        
        return events
    
    def _extract_action_short(self, description: str) -> str:
        """
        Extract just the main action verb from description.
        Used for grouping similar actions.
        
        Examples:
        - "Person TIGHTENS screw with screwdriver" → "TIGHTEN"
        - "Hand PULLS string to turn on light" → "PULL"
        - "Light turns ON" → "LIGHT ON"
        """
        desc_upper = description.upper()
        
        # Action verbs to look for
        action_verbs = [
            'TIGHTEN', 'LOOSEN', 'PULL', 'PUSH', 'TURN', 'TWIST',
            'CONNECT', 'DISCONNECT', 'ATTACH', 'DETACH', 'REMOVE',
            'INSERT', 'EXTRACT', 'POUR', 'MIX', 'CUT', 'SLICE',
            'GRILL', 'COOK', 'ADD', 'ZOOM', 'HOLD', 'GRAB'
        ]
        
        for verb in action_verbs:
            if verb in desc_upper:
                return verb
        
        # State changes
        if 'LIGHT' in desc_upper or 'BULB' in desc_upper:
            if 'ON' in desc_upper or 'SWITCH ON' in desc_upper:
                return 'LIGHT ON'
            elif 'OFF' in desc_upper or 'SWITCH OFF' in desc_upper:
                return 'LIGHT OFF'
        
        # Fallback: return first 2-3 words
        words = description.split()
        if len(words) >= 2:
            return ' '.join(words[:2]).upper()
        return description[:20].upper()
    
    def _is_same_action(self, action1: str, action2: str) -> bool:
        """
        Check if two action descriptions represent the same action.
        Conservative: only return True if very similar.
        """
        if not action1 or not action2:
            return False
        
        # Exact match
        if action1 == action2:
            return True
        
        # Check if one contains the other (for variations)
        a1 = action1.upper().strip()
        a2 = action2.upper().strip()
        
        if a1 in a2 or a2 in a1:
            return True
        
        # Check word overlap (>80% similar)
        words1 = set(a1.split())
        words2 = set(a2.split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        similarity = overlap / total if total > 0 else 0
        
        return similarity > 0.8
    
    def _extract_key_action(self, description: str) -> str:
        """Extract the key action verb from description."""
        desc_lower = description.lower()
        
        # Priority order: most specific to least specific
        if 'loosening' in desc_lower or 'loosen' in desc_lower:
            return "Person LOOSENS screw"
        elif 'tightening' in desc_lower or 'tighten' in desc_lower:
            return "Person TIGHTENS screw"
        elif 'pulling' in desc_lower and 'string' in desc_lower:
            return "Person PULLS string"
        elif 'pushing' in desc_lower and 'wire' in desc_lower:
            return "Person PUSHES wire onto connector"
        elif 'connecting' in desc_lower:
            return "Person CONNECTS wire"
        elif 'disconnecting' in desc_lower:
            return "Person DISCONNECTS wire"
        elif 'removing' in desc_lower:
            return "Person REMOVES component"
        elif 'holding' in desc_lower:
            return "Person HOLDS tool"
        else:
            # Fallback: use first 60 chars of description
            return description[:60] + "..." if len(description) > 60 else description
    
    def _extract_state(self, description: str) -> str:
        """Extract state information (light on/off, etc)."""
        desc_lower = description.lower()
        
        if 'light is on' in desc_lower or 'light on' in desc_lower:
            return "light ON"
        elif 'light is off' in desc_lower or 'light off' in desc_lower:
            return "light OFF"
        elif 'turning on' in desc_lower:
            return "turning ON"
        elif 'turning off' in desc_lower:
            return "turning OFF"
        
        return ""
    
    def _detect_state_change(self, prev_desc: str, curr_desc: str) -> str:
        """Detect state change between consecutive frames."""
        if not prev_desc:
            return ""
        
        prev_lower = prev_desc.lower()
        curr_lower = curr_desc.lower()
        
        # Light state change
        if 'light is off' in prev_lower and 'light is on' in curr_lower:
            return "LIGHT TURNS ON"
        elif 'light is on' in prev_lower and 'light is off' in curr_lower:
            return "LIGHT TURNS OFF"
        
        # Action change
        if 'tightening' in prev_lower and 'loosening' in curr_lower:
            return "SWITCHES TO LOOSENING"
        elif 'loosening' in prev_lower and 'tightening' in curr_lower:
            return "SWITCHES TO TIGHTENING"
        
        return ""
    
    def _build_sequence_summary(self, segments: List[Dict]) -> str:
        """Build a one-line sequence summary."""
        actions = []
        seen = set()
        
        for seg in segments:
            desc = seg.get('description', '').lower()
            
            if 'tighten' in desc and 'tighten' not in seen:
                actions.append("TIGHTEN")
                seen.add('tighten')
            elif 'loosen' in desc and 'loosen' not in seen:
                actions.append("LOOSEN")
                seen.add('loosen')
            elif 'pull' in desc and 'string' in desc and 'pull string' not in seen:
                actions.append("PULL STRING")
                seen.add('pull string')
            elif 'light is on' in desc and 'light on' not in seen:
                actions.append("LIGHT ON")
                seen.add('light on')
        
        if actions:
            return " → ".join(actions)
        return "See steps above"
    
    def _identify_state_changes(self, segments: List[Dict]) -> List[str]:
        """
        Identify critical state changes in the video sequence.
        
        Looks for:
        - Light state changes (OFF→ON, ON→OFF)
        - Action changes (tightening→loosening, connecting→disconnecting)
        - Hand switches (right→left, left→right)
        """
        changes = []
        
        for i in range(len(segments) - 1):
            curr_desc = segments[i].get('description', '').lower()
            next_desc = segments[i+1].get('description', '').lower()
            
            # Check for light state change
            if 'light is off' in curr_desc and 'light is on' in next_desc:
                changes.append("💡 Light turns ON")
            elif 'light is on' in curr_desc and 'light is off' in next_desc:
                changes.append("💡 Light turns OFF")
            
            # Check for action change
            if 'tightening' in curr_desc and 'loosening' in next_desc:
                changes.append("🔧 Action changes: tightening → loosening")
            elif 'loosening' in curr_desc and 'tightening' in next_desc:
                changes.append("🔧 Action changes: loosening → tightening")
            
            # Check for hand switch
            if 'right' in curr_desc and 'left' in next_desc and 'right' not in next_desc:
                changes.append("✋ Hand switches: right → left")
            elif 'left' in curr_desc and 'right' in next_desc and 'left' not in next_desc:
                changes.append("✋ Hand switches: left → right")
        
        return changes[:3]  # Top 3 most important
    
    def _summarize_action_sequence(self, segments: List[Dict]) -> str:
        """
        Create a concise action sequence summary.
        
        Example: "Remove screen → Connect wire → Tighten screw → Turn on light"
        """
        actions = []
        seen_actions = set()
        
        for segment in segments:
            desc = segment.get('description', '').lower()
            
            # Extract key action
            if 'tightening' in desc and 'tightening' not in seen_actions:
                actions.append("Tighten screw")
                seen_actions.add('tightening')
            elif 'loosening' in desc and 'loosening' not in seen_actions:
                actions.append("Loosen screw")
                seen_actions.add('loosening')
            elif 'connecting' in desc and 'connecting' not in seen_actions:
                actions.append("Connect wire")
                seen_actions.add('connecting')
            elif 'pulling' in desc and 'string' in desc and 'pulling string' not in seen_actions:
                actions.append("Pull string")
                seen_actions.add('pulling string')
            elif 'light is on' in desc and 'light on' not in seen_actions:
                actions.append("Light ON")
                seen_actions.add('light on')
            elif 'light is off' in desc and 'light off' not in seen_actions:
                actions.append("Light OFF")
                seen_actions.add('light off')
        
        if actions:
            return " → ".join(actions)
        return ""
    
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
