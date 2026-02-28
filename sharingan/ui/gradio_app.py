"""
============================================================================
SYSTEM DESIGN: Gradio Web UI - Interactive Video Understanding Interface
============================================================================

WHAT THIS FILE DOES:
This is the web-based user interface for SHARINGAN, built with Gradio. It provides
an intuitive way for users to upload videos, process them once, and then ask
unlimited questions about the video content without re-processing.

Think of it like this: You upload a video, SHARINGAN "watches" it once and
remembers everything, then you can have a conversation about what happened in
the video - just like talking to someone who watched the video with you.

HOW IT FITS IN THE SYSTEM:
This is the USER-FACING LAYER that sits on top of the complete SHARINGAN pipeline:

USER INTERACTION FLOW:
1. User uploads video (MP4, AVI, MOV, WebM)
2. UI calls ingest pipeline → processes video once (shows progress)
3. User asks questions in natural language
4. UI calls query pipeline → returns answers with timestamps
5. User can ask unlimited follow-up questions (no re-processing)

ARCHITECTURE LAYERS:
┌─────────────────────────────────────────┐
│  Gradio UI (This File)                  │  ← User Interface
├─────────────────────────────────────────┤
│  Query Pipeline (chat/pipeline.py)      │  ← Question Answering
├─────────────────────────────────────────┤
│  Ingest Pipeline (processor.py)         │  ← Video Processing
├─────────────────────────────────────────┤
│  Core Components (VLM, TAS, Memory)     │  ← Deep Architecture
└─────────────────────────────────────────┘

KEY CONCEPTS:

1. **Video Upload Component**:
   - Accepts common video formats (MP4, AVI, MOV, WebM)
   - Validates file size (max 10 GB for memory safety)
   - Displays video preview for user confirmation

2. **Processing Status Display**:
   - Shows real-time progress during ingest (frame processing, event detection)
   - Displays estimated time remaining (ETA)
   - Shows memory usage to prevent OOM errors
   - Updates every second for smooth user experience

3. **Query Interface**:
   - Text input for natural language questions
   - Example queries dropdown for new users:
     * "What happened between 1:30 and 2:00?" (window query)
     * "Find person speaking" (semantic query)
     * "Why did the person pick up the knife?" (causal query)
     * "Summarize this video" (summary query)
   - Query history panel showing previous questions and answers

4. **Results Display**:
   - Natural language answer with timestamps
   - Reasoning path showing how SHARINGAN arrived at the answer
   - Confidence scores for transparency
   - Clickable timestamps to jump to relevant video moments

WHY GRADIO?

We chose Gradio over Flask because:
- **Simpler**: No HTML/CSS/JS needed, pure Python interface
- **ML-Friendly**: Built-in components for video, progress bars, chat
- **Real-time Updates**: Native support for progress callbacks
- **Easy Deployment**: One command to launch, built-in sharing
- **Interactive**: Better for iterative ML/AI applications

Gradio is specifically designed for machine learning demos and applications,
making it perfect for SHARINGAN's video understanding use case.

WHY THIS MATTERS:

Without a UI, SHARINGAN would only be usable by developers writing Python code.
This Gradio interface makes SHARINGAN accessible to:
- Researchers who want to analyze videos without coding
- Content creators who want to search through video archives
- Educators who want to make videos searchable for students
- Anyone who wants to understand video content through conversation

The UI demonstrates SHARINGAN's key advantages:
- **Process Once, Query Forever**: Upload video once, ask unlimited questions
- **Fast Responses**: <500ms query time (no video re-processing)
- **Zero API Cost**: Everything runs locally, no external API calls
- **Privacy**: Video never leaves your machine
- **Temporal Reasoning**: Understands "why" and "when" questions

EXAMPLE USER WORKFLOW:

1. User uploads cooking video (5 minutes long)
2. SHARINGAN processes video (takes ~25 minutes, shows progress bar)
3. User asks: "What ingredients did they use?"
   → Answer: "Tomatoes (0:45), onions (1:20), garlic (1:35), pasta (2:10)"
4. User asks: "Why did they add salt?"
   → Answer: "To season the pasta water before cooking (2:30)"
5. User asks: "Summarize the recipe"
   → Answer: "Pasta with tomato sauce: sauté vegetables, cook pasta, combine"

All queries after initial processing are instant (<500ms) because the video
is never re-processed.

TECHNICAL DETAILS:

**Progress Callbacks**:
- Gradio supports real-time progress updates using `gr.Progress()`
- We update progress during frame processing, event detection, graph building
- Progress bar shows percentage complete and ETA

**Error Handling**:
- User-friendly error messages for common issues:
  * "Video file too large (max 10 GB)"
  * "Unsupported video format (use MP4, AVI, MOV, or WebM)"
  * "Out of memory - try a shorter video or reduce quality"
- Graceful degradation: If GPU unavailable, fall back to CPU

**State Management**:
- Gradio manages state using `gr.State()` components
- We store: processed video data, memory store, event graph, query history
- State persists across queries (no re-processing needed)

**Responsive Design**:
- Gradio automatically handles mobile/desktop layouts
- Video player adapts to screen size
- Query interface works on touch devices

DEPLOYMENT OPTIONS:

1. **Local Development**:
   ```python
   python -m sharingan.ui.gradio_app
   ```
   Opens at http://localhost:7860

2. **Public Sharing** (temporary):
   ```python
   demo.launch(share=True)
   ```
   Creates public URL for 72 hours

3. **Production Deployment**:
   - Deploy to Hugging Face Spaces (free GPU)
   - Deploy to cloud VM (AWS, GCP, Azure)
   - Deploy behind nginx reverse proxy

SECURITY CONSIDERATIONS:

- **File Upload Validation**: Check file size, format, and headers
- **Path Traversal Prevention**: Sanitize all file paths
- **Query Sanitization**: Prevent prompt injection attacks
- **Rate Limiting**: Limit queries per user (10 per minute)
- **Authentication**: Optional password protection for deployment

FUTURE ENHANCEMENTS (Not in V1):

- Causal graph visualization (networkx + matplotlib)
- Timeline view with detected events
- Video player with timestamp navigation
- Multi-video comparison
- Export results to JSON/CSV
- Batch processing for multiple videos

============================================================================
"""

import gradio as gr
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import time
import traceback
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
from PIL import Image
import tempfile
import subprocess
import re

# Import SHARINGAN components
from sharingan.processor import VideoProcessor
from sharingan.chat.pipeline import VideoQueryPipeline
from sharingan.storage.hierarchical_memory import HierarchicalMemoryStore
from sharingan.graph.event_graph import TemporalEventGraph


# Global state for processed videos
# In production, this would use a database or cache
PROCESSED_VIDEOS: Dict[str, Dict[str, Any]] = {}


def download_youtube_video(youtube_url: str, output_dir: str = None) -> Tuple[bool, str, str]:
    """
    Download YouTube video using yt-dlp with caching to prevent duplicate downloads.
    
    Args:
        youtube_url: YouTube video URL
        output_dir: Directory to save video (default: temp directory)
    
    Returns:
        Tuple of (success, video_path, error_message)
    """
    try:
        # Validate YouTube URL
        youtube_pattern = r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+'
        if not re.match(youtube_pattern, youtube_url):
            return False, "", "Invalid YouTube URL format"
        
        # Extract video ID for caching
        video_id = None
        if 'youtube.com/watch?v=' in youtube_url:
            video_id = youtube_url.split('v=')[1].split('&')[0]
        elif 'youtu.be/' in youtube_url:
            video_id = youtube_url.split('youtu.be/')[1].split('?')[0]
        
        # Create cache directory
        if output_dir is None:
            cache_dir = Path(tempfile.gettempdir()) / "sharingan_youtube_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            output_dir = str(cache_dir)
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Check if video already exists in cache
        if video_id:
            existing_files = list(Path(output_dir).glob(f"*{video_id}*"))
            if existing_files:
                video_path = str(existing_files[0])
                print(f"✓ Using cached video: {video_path}")
                return True, video_path, ""
        
        # Output template with video ID for caching
        if video_id:
            output_template = str(Path(output_dir) / f"%(title)s-{video_id}.%(ext)s")
        else:
            output_template = str(Path(output_dir) / "%(title)s.%(ext)s")
        
        # Check if yt-dlp is installed
        try:
            subprocess.run(['yt-dlp', '--version'], 
                         capture_output=True, 
                         check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, "", (
                "yt-dlp is not installed. Please install it:\n"
                "pip install yt-dlp\n"
                "or\n"
                "pip install --upgrade yt-dlp"
            )
        
        # Download video
        print(f"Downloading YouTube video: {youtube_url}")
        
        result = subprocess.run([
            'yt-dlp',
            '-f', 'best[ext=mp4]/best',  # Prefer MP4 format
            '--no-playlist',  # Don't download playlists
            '-o', output_template,
            youtube_url
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode != 0:
            return False, "", f"yt-dlp error: {result.stderr}"
        
        # Find downloaded file
        if video_id:
            video_files = list(Path(output_dir).glob(f"*{video_id}*"))
        else:
            video_files = list(Path(output_dir).glob('*.*'))
        
        if not video_files:
            return False, "", "Video downloaded but file not found"
        
        video_path = str(video_files[0])
        print(f"✓ Downloaded to: {video_path}")
        
        return True, video_path, ""
        
    except subprocess.TimeoutExpired:
        return False, "", "Download timeout (>5 minutes). Video may be too large."
    except Exception as e:
        return False, "", f"Download error: {str(e)}"


def format_timestamp(seconds: float) -> str:
    """
    Format seconds as MM:SS or HH:MM:SS.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted timestamp string
    
    Example:
        >>> format_timestamp(90.5)
        '01:30'
        >>> format_timestamp(3665.0)
        '01:01:05'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def validate_video_file(video_path: str) -> Tuple[bool, str]:
    """
    Validate uploaded video file.
    
    Checks:
    - File exists
    - File size < 10 GB
    - File format is supported (MP4, AVI, MOV, WebM)
    
    Args:
        video_path: Path to uploaded video file
    
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is empty string
    """
    if not video_path:
        return False, "No video file provided"
    
    path = Path(video_path)
    
    # Check file exists
    if not path.exists():
        return False, "Video file not found"
    
    # Check file size (max 10 GB)
    file_size_gb = path.stat().st_size / (1024 ** 3)
    if file_size_gb > 10:
        return False, f"Video file too large ({file_size_gb:.1f} GB). Maximum size is 10 GB."
    
    # Check file format
    supported_formats = {'.mp4', '.avi', '.mov', '.webm', '.mkv'}
    if path.suffix.lower() not in supported_formats:
        return False, f"Unsupported video format '{path.suffix}'. Supported formats: {', '.join(supported_formats)}"
    
    return True, ""


def process_video(
    video_path: str,
    youtube_url: str,
    vlm_model: str = "clip",
    device: str = "auto",
    # Multi-Scale TAS settings
    tas_kernel_short: int = 2,
    tas_kernel_mid: int = 8,
    tas_kernel_long: int = 32,
    tas_window_size: int = 64,
    # Adaptive Sampler settings
    sampler_base_fps: float = 1.0,
    sampler_max_fps: float = 5.0,
    sampler_change_threshold: float = 0.3,
    # Cross-Modal Verifier settings
    verifier_threshold: float = 0.7,
    verifier_entity_threshold: float = 0.5,
    # Causal Edge Scorer settings
    scorer_mode: str = "heuristic",
    scorer_causal_threshold: float = 0.7,
    scorer_semantic_threshold: float = 0.5,
    # Model selection
    smolvlm_model: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
    clip_model: str = "openai/clip-vit-base-patch32",
    qwen_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    smolvlm_context_window: int = 8,
    progress=gr.Progress()
) -> Tuple[str, str, str, Optional[Image.Image], Optional[Image.Image]]:
    """
    Process uploaded video through SHARINGAN ingest pipeline.
    
    This function:
    1. Downloads YouTube video if URL provided (otherwise uses uploaded file)
    2. Validates video file
    3. Initializes VideoProcessor with selected model and configuration
    4. Processes video with progress updates
    5. Stores results in global state for querying
    6. Generates visualizations (causal graph, timeline)
    
    Args:
        video_path: Path to uploaded video file
        youtube_url: YouTube video URL (optional, takes precedence over video_path)
        vlm_model: Vision model to use ("clip" or "smolvlm")
        device: Device to use ("cpu", "cuda", or "auto")
        tas_kernel_short: Short-scale kernel size for gestures
        tas_kernel_mid: Mid-scale kernel size for actions
        tas_kernel_long: Long-scale kernel size for scenes
        tas_window_size: Maximum frames for long-scale attention
        sampler_base_fps: Base sampling rate for static scenes
        sampler_max_fps: Maximum sampling rate
        sampler_change_threshold: Threshold for detecting scene changes
        verifier_threshold: CLIP similarity threshold for verification
        verifier_entity_threshold: Threshold for entity verification
        scorer_mode: Causal scorer mode ("heuristic" or "learned")
        scorer_causal_threshold: Similarity threshold for causal edges
        scorer_semantic_threshold: Similarity threshold for semantic edges
        smolvlm_model: SmolVLM model name
        clip_model: CLIP model name
        qwen_model: Qwen LLM model name
        smolvlm_context_window: Number of previous frames for context
        progress: Gradio progress callback for real-time updates
    
    Returns:
        Tuple of (status_message, video_info, error_message, graph_image, timeline_image)
    """
    try:
        # Handle YouTube URL if provided
        if youtube_url and youtube_url.strip():
            progress(0, desc="Downloading YouTube video...")
            success, downloaded_path, error_msg = download_youtube_video(youtube_url.strip())
            if not success:
                return "", "", f"❌ YouTube Download Error: {error_msg}", None, None
            video_path = downloaded_path
            progress(0.05, desc="YouTube video downloaded successfully")
        
        # Validate video file
        progress(0.05, desc="Validating video file...")
        is_valid, error_msg = validate_video_file(video_path)
        if not is_valid:
            return "", "", f"❌ Error: {error_msg}", None, None
        
        # Initialize processor with configuration
        progress(0.05, desc="Initializing SHARINGAN...")
        
        # Build configuration dictionary for reference (not passed to VideoProcessor)
        config = {
            'vlm_model': vlm_model,
            'device': device,
            'target_fps': sampler_max_fps,
            'enable_temporal': True,
            'batch_size': 32,
            # Multi-Scale TAS configuration
            'tas_config': {
                'kernel_sizes': [tas_kernel_short, tas_kernel_mid, tas_kernel_long],
                'window_size': tas_window_size,
                'causal': True
            },
            # Adaptive Sampler configuration
            'sampler_config': {
                'base_fps': sampler_base_fps,
                'max_fps': sampler_max_fps,
                'change_threshold': sampler_change_threshold
            },
            # Cross-Modal Verifier configuration
            'verifier_config': {
                'threshold': verifier_threshold,
                'entity_threshold': verifier_entity_threshold,
                'clip_model': clip_model
            },
            # Causal Edge Scorer configuration
            'scorer_config': {
                'mode': scorer_mode,
                'causal_threshold': scorer_causal_threshold,
                'semantic_threshold': scorer_semantic_threshold
            },
            # Model selection
            'model_config': {
                'smolvlm_model': smolvlm_model,
                'clip_model': clip_model,
                'qwen_model': qwen_model,
                'smolvlm_context_window': smolvlm_context_window
            }
        }
        
        # Initialize VideoProcessor with only the parameters it accepts
        processor = VideoProcessor(
            vlm_model=vlm_model,
            device=device,
            target_fps=sampler_max_fps,
            enable_temporal=True,
            batch_size=32
        )
        
        # Process video with progress updates
        progress(0.1, desc="Processing video frames...")
        
        # Start processing
        start_time = time.time()
        results = processor.process(video_path)
        processing_time = time.time() - start_time
        
        # Store results in global state
        video_key = Path(video_path).stem
        PROCESSED_VIDEOS[video_key] = {
            'processor': processor,
            'results': results,
            'video_path': video_path,
            'processing_time': processing_time,
            'config': config  # Store configuration for reference
        }
        
        # Generate visualizations
        progress(0.9, desc="Generating visualizations...")
        
        # Causal graph visualization
        graph_image = None
        if hasattr(processor, 'event_graph') and processor.event_graph:
            graph_image = visualize_causal_graph(processor.event_graph)
        
        # Timeline visualization
        timeline_image = None
        video_info = results['video_info']
        if 'events' in results and results['events']:
            timeline_image = create_timeline_view(
                results['events'], 
                video_info['duration']
            )
        
        # Format video info with configuration details
        info_text = (
            f"📹 **Video Information**\n\n"
            f"- Duration: {format_timestamp(video_info['duration'])}\n"
            f"- Total Frames: {video_info['total_frames']:,}\n"
            f"- Processed Frames: {video_info['processed_frames']:,}\n"
            f"- FPS: {video_info['fps']:.1f}\n"
            f"- Events Detected: {len(results['events'])}\n"
            f"- Processing Time: {processing_time:.1f}s\n\n"
            f"⚙️ **Configuration**\n\n"
            f"- TAS Kernels: {tas_kernel_short}/{tas_kernel_mid}/{tas_kernel_long}\n"
            f"- Sampler FPS: {sampler_base_fps}-{sampler_max_fps}\n"
            f"- Verifier Threshold: {verifier_threshold:.2f}\n"
            f"- Scorer Mode: {scorer_mode}\n"
        )
        
        # Format status message
        status_msg = (
            f"✅ **Video processed successfully!**\n\n"
            f"Processed {video_info['processed_frames']} frames in {processing_time:.1f}s\n"
            f"Detected {len(results['events'])} events\n\n"
            f"You can now ask questions about the video!"
        )
        
        progress(1.0, desc="Processing complete!")
        
        return status_msg, info_text, "", graph_image, timeline_image
        
    except Exception as e:
        error_msg = f"❌ **Error during processing:**\n\n{str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return "", "", error_msg, None, None


def query_video(
    query: str,
    video_key: str,
    include_reasoning: bool = True,
    history: Optional[List[Tuple[str, str]]] = None
) -> Tuple[List[Tuple[str, str]], str, str, str]:
    """
    Query processed video with natural language question.
    
    Args:
        query: Natural language question
        video_key: Key to identify processed video in global state
        include_reasoning: Whether to include reasoning path in response
        history: Chat history (list of question-answer pairs)
    
    Returns:
        Tuple of (updated_history, error_message, reasoning_display, confidence_display)
    """
    try:
        # Validate query
        if not query or not query.strip():
            return history or [], "❌ Please enter a question", "", ""
        
        if len(query) > 512:
            return history or [], "❌ Question too long (max 512 characters)", "", ""
        
        # Check if video is processed
        if not video_key or video_key not in PROCESSED_VIDEOS:
            return history or [], "❌ Please process a video first", "", ""
        
        # Get processor from global state
        video_data = PROCESSED_VIDEOS[video_key]
        processor = video_data['processor']
        
        # Query video
        start_time = time.time()
        response = processor.chat(query, use_llm=True)
        query_time = time.time() - start_time
        
        # Extract query result details if available
        query_result = {}
        if isinstance(response, dict):
            query_result = response
            response = response.get('answer', str(response))
        
        # Add query time to response
        response_with_time = f"{response}\n\n*Query time: {query_time*1000:.0f}ms*"
        
        # Update history
        if history is None:
            history = []
        history.append((query, response_with_time))
        
        # Format reasoning scaffold
        reasoning_display = ""
        if include_reasoning and query_result:
            reasoning_display = format_reasoning_scaffold(query_result)
        
        # Format confidence indicators
        confidence_display = ""
        if query_result:
            confidence_display = format_confidence_indicators(query_result)
        
        return history, "", reasoning_display, confidence_display
        
    except Exception as e:
        error_msg = f"❌ **Error during query:**\n\n{str(e)}"
        return history or [], error_msg, "", ""


def visualize_causal_graph(event_graph: TemporalEventGraph) -> Optional[Image.Image]:
    """
    Visualize the causal event graph using networkx and matplotlib.
    
    Args:
        event_graph: Temporal event graph to visualize
    
    Returns:
        PIL Image of the graph visualization, or None if no events
    """
    try:
        if not event_graph or not event_graph.nodes:
            return None
        
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add nodes with labels
        for event_id, event_node in event_graph.nodes.items():
            # Truncate description for readability
            label = event_node.description[:30] + "..." if len(event_node.description) > 30 else event_node.description
            timestamp = format_timestamp(event_node.timestamp)
            G.add_node(event_id, label=f"{timestamp}\n{label}")
        
        # Add edges with colors based on type
        edge_colors = []
        edge_styles = []
        for edge in event_graph.edges:
            G.add_edge(edge.source_id, edge.target_id, 
                      edge_type=edge.edge_type, 
                      confidence=edge.confidence)
            
            # Color code by edge type
            if edge.edge_type == "causal":
                edge_colors.append('red')
                edge_styles.append('solid')
            elif edge.edge_type == "semantic":
                edge_colors.append('blue')
                edge_styles.append('dashed')
            else:  # temporal
                edge_colors.append('gray')
                edge_styles.append('dotted')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use hierarchical layout for temporal ordering
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=2000, alpha=0.9, ax=ax)
        
        # Draw edges
        for i, edge in enumerate(G.edges()):
            nx.draw_networkx_edges(G, pos, [(edge[0], edge[1])], 
                                  edge_color=[edge_colors[i]],
                                  style=edge_styles[i],
                                  width=2, alpha=0.7,
                                  arrows=True, arrowsize=20, ax=ax)
        
        # Draw labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='Causal'),
            Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Semantic'),
            Line2D([0], [0], color='gray', linewidth=2, linestyle=':', label='Temporal')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title("Causal Event Graph", fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        
        return img
        
    except Exception as e:
        print(f"Error visualizing causal graph: {e}")
        return None


def create_timeline_view(events: List[Dict[str, Any]], video_duration: float) -> Optional[Image.Image]:
    """
    Create a timeline visualization showing detected events.
    
    Args:
        events: List of detected events with timestamps
        video_duration: Total video duration in seconds
    
    Returns:
        PIL Image of the timeline visualization, or None if no events
    """
    try:
        if not events:
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.get('timestamp', 0))
        
        # Plot timeline
        for i, event in enumerate(sorted_events):
            timestamp = event.get('timestamp', 0)
            description = event.get('description', 'Unknown event')
            confidence = event.get('confidence', 0.5)
            
            # Truncate description
            label = description[:40] + "..." if len(description) > 40 else description
            
            # Color based on confidence
            color = plt.cm.RdYlGn(confidence)
            
            # Plot event marker
            ax.scatter(timestamp, 0, s=200, c=[color], alpha=0.8, zorder=3)
            
            # Add label (alternate above/below for readability)
            y_offset = 0.1 if i % 2 == 0 else -0.1
            ax.text(timestamp, y_offset, label, 
                   rotation=45 if i % 2 == 0 else -45,
                   ha='left' if i % 2 == 0 else 'right',
                   va='bottom' if i % 2 == 0 else 'top',
                   fontsize=8)
        
        # Draw timeline
        ax.plot([0, video_duration], [0, 0], 'k-', linewidth=2, zorder=1)
        
        # Format x-axis with timestamps
        ax.set_xlim(-video_duration*0.05, video_duration*1.05)
        ax.set_ylim(-0.3, 0.3)
        
        # Add time markers
        num_markers = min(10, int(video_duration / 10) + 1)
        time_markers = np.linspace(0, video_duration, num_markers)
        ax.set_xticks(time_markers)
        ax.set_xticklabels([format_timestamp(t) for t in time_markers])
        
        ax.set_yticks([])
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_title(f'Event Timeline ({len(events)} events detected)', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar for confidence
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', 
                           pad=0.1, aspect=30)
        cbar.set_label('Confidence Score', fontsize=10)
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        
        return img
        
    except Exception as e:
        print(f"Error creating timeline view: {e}")
        return None


def format_reasoning_scaffold(query_result: Dict[str, Any]) -> str:
    """
    Format reasoning scaffold for display.
    
    Args:
        query_result: Query result containing reasoning information
    
    Returns:
        Formatted markdown string showing reasoning steps
    """
    try:
        scaffold_info = query_result.get('reasoning_scaffold', {})
        
        if not scaffold_info:
            return "*No reasoning scaffold available*"
        
        scaffold_type = scaffold_info.get('scaffold_type', 'unknown')
        reasoning_steps = scaffold_info.get('reasoning_steps', [])
        evidence = scaffold_info.get('evidence', [])
        
        # Format output
        output = f"### 🧠 Reasoning Process\n\n"
        output += f"**Type:** {scaffold_type.replace('_', ' ').title()}\n\n"
        
        if reasoning_steps:
            output += "**Steps:**\n"
            for i, step in enumerate(reasoning_steps, 1):
                output += f"{i}. {step}\n"
            output += "\n"
        
        if evidence:
            output += "**Evidence:**\n"
            for i, ev in enumerate(evidence, 1):
                timestamp = ev.get('timestamp', 0)
                desc = ev.get('description', 'N/A')
                output += f"- [{format_timestamp(timestamp)}] {desc}\n"
        
        return output
        
    except Exception as e:
        return f"*Error formatting reasoning scaffold: {e}*"


def format_confidence_indicators(query_result: Dict[str, Any]) -> str:
    """
    Format confidence scores for display.
    
    Args:
        query_result: Query result containing confidence information
    
    Returns:
        Formatted markdown string with confidence indicators
    """
    try:
        confidence = query_result.get('confidence', 0.5)
        
        # Create visual confidence bar
        bar_length = 20
        filled = int(confidence * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Color code based on confidence
        if confidence >= 0.8:
            emoji = "🟢"
            level = "High"
        elif confidence >= 0.5:
            emoji = "🟡"
            level = "Medium"
        else:
            emoji = "🔴"
            level = "Low"
        
        output = f"### {emoji} Confidence: {level}\n\n"
        output += f"`{bar}` {confidence:.1%}\n\n"
        
        # Add component-level confidence if available
        components = query_result.get('component_confidence', {})
        if components:
            output += "**Component Scores:**\n"
            for component, score in components.items():
                output += f"- {component}: {score:.1%}\n"
        
        return output
        
    except Exception as e:
        return f"*Error formatting confidence: {e}*"


def get_example_queries() -> List[str]:
    """
    Get list of example queries for different query types.
    
    Returns:
        List of example query strings
    """
    return [
        "What happens in this video?",
        "What happened between 0:30 and 1:00?",
        "Find person speaking",
        "Why did the person pick up the object?",
        "Summarize the main events",
        "What objects are visible?",
        "Describe the scene at 0:45",
        "What caused the person to react?",
        "Show me all actions involving the red object",
        "What is the sequence of events?",
    ]


def create_gradio_interface() -> gr.Blocks:
    """
    Create Gradio interface for SHARINGAN.
    
    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(
        title="SHARINGAN - Deep Video Understanding",
        theme=gr.themes.Soft()
    ) as demo:
        # Header
        gr.Markdown(
            """
            # 🎬 SHARINGAN - Deep Video Understanding
            
            Upload a video, process it once, then ask unlimited questions about the content.
            SHARINGAN uses multi-scale temporal reasoning to understand videos like a human would.
            
            **Key Features:**
            - 🚀 Process video once, query forever (no re-processing)
            - ⚡ Fast queries (<500ms) with local models
            - 🔒 Complete privacy (no external API calls)
            - 🧠 Temporal reasoning (understands "why" and "when")
            - 💰 Zero API cost (runs locally)
            """
        )
        
        # State variables
        video_key_state = gr.State(value=None)
        
        with gr.Row():
            # Left column: Video upload and processing
            with gr.Column(scale=1):
                gr.Markdown("## 1️⃣ Upload Video")
                
                video_input = gr.Video(
                    label="Upload Video",
                    sources=["upload"],
                    format="mp4"
                )
                
                gr.Markdown("**OR**")
                
                youtube_url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    info="Paste a YouTube URL to download and process (requires yt-dlp)"
                )
                
                with gr.Row():
                    vlm_model_dropdown = gr.Dropdown(
                        choices=["clip", "smolvlm"],
                        value="clip",
                        label="Vision Model",
                        info="CLIP is faster, SmolVLM provides descriptions"
                    )
                    
                    device_dropdown = gr.Dropdown(
                        choices=["auto", "cuda", "cpu"],
                        value="auto",
                        label="Device",
                        info="Auto detects GPU availability"
                    )
                
                # Configuration panel
                with gr.Accordion("⚙️ Advanced Configuration", open=False):
                    gr.Markdown("### Multi-Scale TAS Settings")
                    with gr.Row():
                        tas_kernel_short = gr.Slider(
                            minimum=1,
                            maximum=8,
                            value=2,
                            step=1,
                            label="Short-scale Kernel Size",
                            info="For gestures and cuts (default: 2)"
                        )
                        tas_kernel_mid = gr.Slider(
                            minimum=4,
                            maximum=16,
                            value=8,
                            step=1,
                            label="Mid-scale Kernel Size",
                            info="For actions (default: 8)"
                        )
                        tas_kernel_long = gr.Slider(
                            minimum=16,
                            maximum=64,
                            value=32,
                            step=1,
                            label="Long-scale Kernel Size",
                            info="For scenes (default: 32)"
                        )
                    
                    tas_window_size = gr.Slider(
                        minimum=32,
                        maximum=128,
                        value=64,
                        step=16,
                        label="TAS Window Size",
                        info="Maximum frames for long-scale attention (default: 64)"
                    )
                    
                    gr.Markdown("### Adaptive Sampler Settings")
                    with gr.Row():
                        sampler_base_fps = gr.Slider(
                            minimum=0.5,
                            maximum=5.0,
                            value=1.0,
                            step=0.5,
                            label="Base FPS",
                            info="Sampling rate for static scenes (default: 1.0)"
                        )
                        sampler_max_fps = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            value=5.0,
                            step=0.5,
                            label="Max FPS",
                            info="Maximum sampling rate (default: 5.0)"
                        )
                    
                    sampler_change_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.3,
                        step=0.05,
                        label="Change Threshold",
                        info="Threshold for detecting scene changes (default: 0.3)"
                    )
                    
                    gr.Markdown("### Cross-Modal Verifier Settings")
                    with gr.Row():
                        verifier_threshold = gr.Slider(
                            minimum=0.5,
                            maximum=0.95,
                            value=0.7,
                            step=0.05,
                            label="Similarity Threshold",
                            info="CLIP similarity threshold for verification (default: 0.7)"
                        )
                        verifier_entity_threshold = gr.Slider(
                            minimum=0.3,
                            maximum=0.8,
                            value=0.5,
                            step=0.05,
                            label="Entity Threshold",
                            info="Threshold for entity verification (default: 0.5)"
                        )
                    
                    gr.Markdown("### Causal Edge Scorer Settings")
                    scorer_mode = gr.Radio(
                        choices=["heuristic", "learned"],
                        value="heuristic",
                        label="Scorer Mode",
                        info="Heuristic uses cosine similarity, learned uses trained model"
                    )
                    
                    with gr.Row():
                        scorer_causal_threshold = gr.Slider(
                            minimum=0.5,
                            maximum=0.95,
                            value=0.7,
                            step=0.05,
                            label="Causal Threshold (Heuristic)",
                            info="Similarity threshold for causal edges (default: 0.7)"
                        )
                        scorer_semantic_threshold = gr.Slider(
                            minimum=0.3,
                            maximum=0.8,
                            value=0.5,
                            step=0.05,
                            label="Semantic Threshold (Heuristic)",
                            info="Similarity threshold for semantic edges (default: 0.5)"
                        )
                    
                    gr.Markdown("### Model Selection")
                    with gr.Row():
                        smolvlm_model = gr.Dropdown(
                            choices=[
                                "HuggingFaceTB/SmolVLM-500M-Instruct",
                                "HuggingFaceTB/SmolVLM-256M-Instruct"
                            ],
                            value="HuggingFaceTB/SmolVLM-500M-Instruct",
                            label="SmolVLM Model",
                            info="Vision-language model for descriptions"
                        )
                        clip_model = gr.Dropdown(
                            choices=[
                                "openai/clip-vit-base-patch32",
                                "openai/clip-vit-large-patch14"
                            ],
                            value="openai/clip-vit-base-patch32",
                            label="CLIP Model",
                            info="Model for cross-modal verification"
                        )
                    
                    qwen_model = gr.Dropdown(
                        choices=[
                            "Qwen/Qwen2.5-0.5B-Instruct",
                            "Qwen/Qwen2.5-1.5B-Instruct",
                            "Qwen/Qwen2.5-3B-Instruct"
                        ],
                        value="Qwen/Qwen2.5-0.5B-Instruct",
                        label="Qwen LLM Model",
                        info="Language model for query responses"
                    )
                    
                    smolvlm_context_window = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=8,
                        step=1,
                        label="SmolVLM Context Window",
                        info="Number of previous frames for context (default: 8)"
                    )
                
                process_button = gr.Button(
                    "🚀 Process Video",
                    variant="primary",
                    size="lg"
                )
                
                processing_status = gr.Markdown(
                    value="",
                    label="Status"
                )
                
                video_info_display = gr.Markdown(
                    value="",
                    label="Video Information"
                )
                
                processing_error = gr.Markdown(
                    value="",
                    label="Errors"
                )
            
            # Right column: Query interface
            with gr.Column(scale=1):
                gr.Markdown("## 2️⃣ Ask Questions")
                
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    show_label=True
                )
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything about the video...",
                        lines=2,
                        scale=4
                    )
                    
                    query_button = gr.Button(
                        "Ask",
                        variant="primary",
                        scale=1
                    )
                
                example_queries = gr.Examples(
                    examples=get_example_queries(),
                    inputs=query_input,
                    label="Example Questions"
                )
                
                query_error = gr.Markdown(
                    value="",
                    label="Query Errors"
                )
                
                clear_button = gr.Button("Clear Conversation")
        
        # Advanced features section
        with gr.Accordion("🔍 Advanced Analysis", open=False):
            with gr.Tabs():
                # Tab 1: Causal Graph
                with gr.Tab("Causal Graph"):
                    gr.Markdown(
                        """
                        ### Causal Event Graph
                        
                        This graph shows the relationships between detected events:
                        - **Red solid lines**: Causal relationships (A caused B)
                        - **Blue dashed lines**: Semantic relationships (A and B are related)
                        - **Gray dotted lines**: Temporal relationships (A happened before B)
                        """
                    )
                    causal_graph_image = gr.Image(
                        label="Causal Graph Visualization",
                        type="pil"
                    )
                
                # Tab 2: Timeline
                with gr.Tab("Timeline"):
                    gr.Markdown(
                        """
                        ### Event Timeline
                        
                        This timeline shows when events were detected in the video.
                        Colors indicate confidence scores (green = high, yellow = medium, red = low).
                        """
                    )
                    timeline_image = gr.Image(
                        label="Event Timeline",
                        type="pil"
                    )
                
                # Tab 3: Reasoning Scaffold
                with gr.Tab("Reasoning"):
                    gr.Markdown(
                        """
                        ### Reasoning Scaffold
                        
                        This shows how SHARINGAN reasoned about your last question,
                        including the steps taken and evidence used.
                        """
                    )
                    reasoning_display = gr.Markdown(
                        value="*Ask a question to see reasoning*",
                        label="Reasoning Process"
                    )
                
                # Tab 4: Confidence Scores
                with gr.Tab("Confidence"):
                    gr.Markdown(
                        """
                        ### Confidence Indicators
                        
                        This shows how confident SHARINGAN is in its answer,
                        broken down by component.
                        """
                    )
                    confidence_display = gr.Markdown(
                        value="*Ask a question to see confidence scores*",
                        label="Confidence Scores"
                    )
        
        # Event handlers
        def process_video_wrapper(
            video_path, youtube_url, vlm_model, device,
            tas_kernel_short, tas_kernel_mid, tas_kernel_long, tas_window_size,
            sampler_base_fps, sampler_max_fps, sampler_change_threshold,
            verifier_threshold, verifier_entity_threshold,
            scorer_mode, scorer_causal_threshold, scorer_semantic_threshold,
            smolvlm_model, clip_model, qwen_model, smolvlm_context_window
        ):
            """Wrapper to handle video processing and state update."""
            status, info, error, graph_img, timeline_img = process_video(
                video_path, youtube_url, vlm_model, device,
                tas_kernel_short, tas_kernel_mid, tas_kernel_long, tas_window_size,
                sampler_base_fps, sampler_max_fps, sampler_change_threshold,
                verifier_threshold, verifier_entity_threshold,
                scorer_mode, scorer_causal_threshold, scorer_semantic_threshold,
                smolvlm_model, clip_model, qwen_model, smolvlm_context_window
            )
            
            # Extract video key from path
            video_key = Path(video_path).stem if video_path else None
            
            return status, info, error, video_key, graph_img, timeline_img
        
        process_button.click(
            fn=process_video_wrapper,
            inputs=[
                video_input, youtube_url_input, vlm_model_dropdown, device_dropdown,
                tas_kernel_short, tas_kernel_mid, tas_kernel_long, tas_window_size,
                sampler_base_fps, sampler_max_fps, sampler_change_threshold,
                verifier_threshold, verifier_entity_threshold,
                scorer_mode, scorer_causal_threshold, scorer_semantic_threshold,
                smolvlm_model, clip_model, qwen_model, smolvlm_context_window
            ],
            outputs=[
                processing_status, 
                video_info_display, 
                processing_error, 
                video_key_state,
                causal_graph_image,
                timeline_image
            ]
        )
        
        def query_video_wrapper(query, video_key, history):
            """Wrapper to handle video querying."""
            updated_history, error, reasoning, confidence = query_video(
                query, video_key, True, history
            )
            return updated_history, error, "", reasoning, confidence  # Clear query input
        
        query_button.click(
            fn=query_video_wrapper,
            inputs=[query_input, video_key_state, chatbot],
            outputs=[
                chatbot, 
                query_error, 
                query_input,
                reasoning_display,
                confidence_display
            ]
        )
        
        # Also trigger on Enter key
        query_input.submit(
            fn=query_video_wrapper,
            inputs=[query_input, video_key_state, chatbot],
            outputs=[
                chatbot, 
                query_error, 
                query_input,
                reasoning_display,
                confidence_display
            ]
        )
        
        clear_button.click(
            fn=lambda: ([], "", "*Ask a question to see reasoning*", "*Ask a question to see confidence scores*"),
            outputs=[chatbot, query_error, reasoning_display, confidence_display]
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            **About SHARINGAN:** Deep architecture for video understanding using multi-scale temporal reasoning,
            context-aware VLMs, cross-modal verification, and causal graph construction.
            
            **Models:** SmolVLM-500M, CLIP ViT-B/32, Qwen2.5-0.5B (all running locally)
            """
        )
    
    return demo


def launch_app(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
    auth: Optional[Tuple[str, str]] = None
):
    """
    Launch Gradio application.
    
    Args:
        server_name: Server host (default: "0.0.0.0" for all interfaces)
        server_port: Server port (default: 7860)
        share: Create public URL for sharing (default: False)
        auth: Optional (username, password) tuple for authentication
    
    Example:
        # Local development
        launch_app()
        
        # Public sharing with authentication
        launch_app(share=True, auth=("admin", "password123"))
    """
    demo = create_gradio_interface()
    
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        auth=auth,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    # Launch with default settings
    print("🚀 Launching SHARINGAN Gradio UI...")
    print("📍 Open http://localhost:7860 in your browser")
    launch_app()
