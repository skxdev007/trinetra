"""
Visualization tools for temporal processing pipeline.

This module provides tools to visualize:
1. TAS (Temporal Attention Shift) outputs at different scales
2. GRU hidden states over time
3. Temporal event graph (TEG) structure
4. Final context sent to LLM
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


class TemporalVisualizer:
    """Visualize temporal processing pipeline outputs."""
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def visualize_tas_outputs(
        self,
        timestamps: List[float],
        raw_embeddings: np.ndarray,
        tas_short: np.ndarray,
        tas_mid: np.ndarray,
        tas_long: np.ndarray,
        gru_output: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Visualize TAS outputs at different temporal scales.
        
        Args:
            timestamps: Frame timestamps
            raw_embeddings: Raw frame embeddings (N, D)
            tas_short: Short-scale TAS output (N, D)
            tas_mid: Mid-scale TAS output (N, D)
            tas_long: Long-scale TAS output (N, D)
            gru_output: GRU output (N, D)
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(5, 1, figsize=(16, 12))
        fig.suptitle('Temporal Processing Pipeline - Multi-Scale TAS + GRU', fontsize=16, fontweight='bold')
        
        # Compute temporal derivatives (rate of change)
        def compute_change(embeddings):
            """Compute L2 norm of frame-to-frame differences."""
            if len(embeddings) < 2:
                return np.zeros(len(embeddings))
            diffs = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
            return np.concatenate([[0], diffs])  # Prepend 0 for first frame
        
        raw_change = compute_change(raw_embeddings)
        short_change = compute_change(tas_short)
        mid_change = compute_change(tas_mid)
        long_change = compute_change(tas_long)
        gru_change = compute_change(gru_output)
        
        # Plot 1: Raw embeddings
        axes[0].plot(timestamps, raw_change, color='gray', linewidth=1.5, alpha=0.7)
        axes[0].fill_between(timestamps, 0, raw_change, color='gray', alpha=0.3)
        axes[0].set_ylabel('Change\nMagnitude', fontsize=10, fontweight='bold')
        axes[0].set_title('Raw Frame Embeddings (No Temporal Context)', fontsize=12, loc='left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(timestamps[0], timestamps[-1])
        
        # Plot 2: Short-scale TAS (gestures, 2-frame kernel)
        axes[1].plot(timestamps, short_change, color='#FF6B6B', linewidth=1.5, alpha=0.7)
        axes[1].fill_between(timestamps, 0, short_change, color='#FF6B6B', alpha=0.3)
        axes[1].set_ylabel('Change\nMagnitude', fontsize=10, fontweight='bold')
        axes[1].set_title('Short-Scale TAS (Kernel=2, Gestures & Quick Actions)', fontsize=12, loc='left')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(timestamps[0], timestamps[-1])
        
        # Plot 3: Mid-scale TAS (actions, 8-frame kernel)
        axes[2].plot(timestamps, mid_change, color='#4ECDC4', linewidth=1.5, alpha=0.7)
        axes[2].fill_between(timestamps, 0, mid_change, color='#4ECDC4', alpha=0.3)
        axes[2].set_ylabel('Change\nMagnitude', fontsize=10, fontweight='bold')
        axes[2].set_title('Mid-Scale TAS (Kernel=8, Actions & Movements)', fontsize=12, loc='left')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(timestamps[0], timestamps[-1])
        
        # Plot 4: Long-scale TAS (scenes, 32-frame kernel)
        axes[3].plot(timestamps, long_change, color='#95E1D3', linewidth=1.5, alpha=0.7)
        axes[3].fill_between(timestamps, 0, long_change, color='#95E1D3', alpha=0.3)
        axes[3].set_ylabel('Change\nMagnitude', fontsize=10, fontweight='bold')
        axes[3].set_title('Long-Scale TAS (Kernel=32, Scenes & Context)', fontsize=12, loc='left')
        axes[3].grid(True, alpha=0.3)
        axes[3].set_xlim(timestamps[0], timestamps[-1])
        
        # Plot 5: GRU output (full-video memory)
        axes[4].plot(timestamps, gru_change, color='#F38181', linewidth=2, alpha=0.8)
        axes[4].fill_between(timestamps, 0, gru_change, color='#F38181', alpha=0.3)
        axes[4].set_ylabel('Change\nMagnitude', fontsize=10, fontweight='bold')
        axes[4].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        axes[4].set_title('GRU Output (Full-Video Memory + Temporal Context)', fontsize=12, loc='left')
        axes[4].grid(True, alpha=0.3)
        axes[4].set_xlim(timestamps[0], timestamps[-1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved TAS visualization to {save_path}")
        else:
            default_path = self.output_dir / "tas_pipeline.png"
            plt.savefig(default_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved TAS visualization to {default_path}")
        
        plt.close()
    
    def visualize_event_graph(
        self,
        events: List[Dict[str, Any]],
        timestamps: List[float],
        video_duration: float,
        save_path: Optional[str] = None
    ):
        """
        Visualize temporal event graph (TEG).
        
        Args:
            events: List of detected events with timestamps
            timestamps: All frame timestamps
            video_duration: Total video duration
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(16, 8))
        fig.suptitle('Temporal Event Graph (TEG)', fontsize=16, fontweight='bold')
        
        # Plot timeline
        ax.axhline(y=0, color='black', linewidth=2, alpha=0.3)
        
        # Plot all frames as background
        frame_density = np.histogram(timestamps, bins=100)[0]
        bin_edges = np.linspace(0, video_duration, 101)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, frame_density, width=video_duration/100, 
               color='lightgray', alpha=0.3, label='Frame Density')
        
        # Plot events
        event_types = set(e.get('type', 'unknown') for e in events)
        colors = plt.cm.Set3(np.linspace(0, 1, len(event_types)))
        type_to_color = dict(zip(event_types, colors))
        
        for event in events:
            timestamp = event.get('timestamp', 0)
            event_type = event.get('type', 'unknown')
            confidence = event.get('confidence', 0.5)
            
            # Plot event marker
            ax.scatter(timestamp, confidence * 100, 
                      color=type_to_color[event_type],
                      s=100, alpha=0.7, edgecolors='black', linewidth=1)
            
            # Add vertical line
            ax.axvline(x=timestamp, color=type_to_color[event_type], 
                      alpha=0.2, linestyle='--', linewidth=1)
        
        # Legend
        legend_elements = [mpatches.Patch(color=type_to_color[t], label=t) 
                          for t in event_types]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Event Confidence (%)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, video_duration)
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved event graph to {save_path}")
        else:
            default_path = self.output_dir / "event_graph.png"
            plt.savefig(default_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved event graph to {default_path}")
        
        plt.close()
    
    def visualize_llm_context(
        self,
        query: str,
        retrieved_events: List[Dict[str, Any]],
        llm_prompt: str,
        llm_response: str,
        video_duration: float,
        save_path: Optional[str] = None
    ):
        """
        Visualize the context sent to LLM and its response.
        
        Args:
            query: User query
            retrieved_events: Events retrieved for this query
            llm_prompt: Full prompt sent to LLM
            llm_response: LLM's response
            video_duration: Total video duration
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 1], hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'LLM Context Visualization\nQuery: "{query}"', 
                    fontsize=14, fontweight='bold')
        
        # Top left: Query info
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        query_text = f"USER QUERY:\n{query}\n\n"
        query_text += f"RETRIEVED EVENTS: {len(retrieved_events)}\n"
        query_text += f"VIDEO DURATION: {video_duration:.1f}s ({video_duration/60:.1f} min)"
        ax1.text(0.05, 0.5, query_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Middle left: Timeline with retrieved events
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_title('Retrieved Events Timeline', fontsize=12, fontweight='bold')
        
        # Plot timeline
        ax2.axhline(y=0, color='black', linewidth=2, alpha=0.3)
        
        for i, event in enumerate(retrieved_events):
            timestamp = event.get('timestamp', 0)
            confidence = event.get('confidence', 0.5)
            
            # Plot event
            ax2.scatter(timestamp, i, s=200, c=[confidence], 
                       cmap='RdYlGn', vmin=0, vmax=1,
                       edgecolors='black', linewidth=1.5)
            
            # Add timestamp label
            time_str = f"{int(timestamp//60):02d}:{int(timestamp%60):02d}"
            ax2.text(timestamp, i, f"  {time_str}", 
                    fontsize=9, verticalalignment='center')
        
        ax2.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Event Rank', fontsize=10, fontweight='bold')
        ax2.set_xlim(0, video_duration)
        ax2.set_ylim(-1, len(retrieved_events))
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', 
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, orientation='vertical', pad=0.02)
        cbar.set_label('Confidence', fontsize=9)
        
        # Middle right: LLM prompt (truncated)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        ax3.set_title('LLM Prompt (Truncated)', fontsize=12, fontweight='bold')
        
        # Truncate prompt if too long
        max_chars = 800
        prompt_display = llm_prompt[:max_chars]
        if len(llm_prompt) > max_chars:
            prompt_display += f"\n\n... (truncated, {len(llm_prompt)} total chars)"
        
        ax3.text(0.05, 0.95, prompt_display, fontsize=8, 
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3),
                wrap=True)
        
        # Bottom: LLM response
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        ax4.set_title('LLM Response', fontsize=12, fontweight='bold')
        
        response_text = f"{llm_response}"
        ax4.text(0.05, 0.5, response_text, fontsize=10, 
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved LLM context visualization to {save_path}")
        else:
            default_path = self.output_dir / f"llm_context_{hash(query) % 10000}.png"
            plt.savefig(default_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved LLM context visualization to {default_path}")
        
        plt.close()
    
    def visualize_query_comparison(
        self,
        queries: List[Dict[str, Any]],
        video_duration: float,
        save_path: Optional[str] = None
    ):
        """
        Visualize multiple queries and their results side-by-side.
        
        Args:
            queries: List of query results with format:
                {
                    'query': str,
                    'ground_truth': str,
                    'results': List[Dict],
                    'correct': bool
                }
            video_duration: Total video duration
            save_path: Path to save figure
        """
        n_queries = len(queries)
        fig, axes = plt.subplots(n_queries, 1, figsize=(16, 3*n_queries))
        
        if n_queries == 1:
            axes = [axes]
        
        fig.suptitle('Query Results Comparison', fontsize=16, fontweight='bold')
        
        for i, (ax, query_data) in enumerate(zip(axes, queries)):
            query = query_data['query']
            ground_truth = query_data.get('ground_truth', 'Unknown')
            results = query_data.get('results', [])
            correct = query_data.get('correct', False)
            
            # Plot timeline
            ax.axhline(y=0, color='black', linewidth=2, alpha=0.3)
            
            # Plot ground truth if available
            if ground_truth != 'Unknown' and ':' in ground_truth:
                gt_parts = ground_truth.split(':')
                gt_seconds = int(gt_parts[0]) * 60 + int(gt_parts[1])
                ax.axvline(x=gt_seconds, color='green', linewidth=3, 
                          alpha=0.5, linestyle='--', label='Ground Truth')
            
            # Plot results
            for rank, result in enumerate(results[:5], 1):
                timestamp = result.get('timestamp', 0)
                confidence = result.get('confidence', 0)
                
                color = 'green' if rank == 1 and correct else 'red' if rank == 1 else 'orange'
                marker = 'o' if rank == 1 else 's'
                size = 200 if rank == 1 else 100
                
                ax.scatter(timestamp, rank, s=size, c=color, 
                          marker=marker, alpha=0.7, 
                          edgecolors='black', linewidth=1.5)
                
                # Add timestamp label
                time_str = f"{int(timestamp//60):02d}:{int(timestamp%60):02d}"
                ax.text(timestamp, rank, f"  {time_str}\n  ({confidence:.2f})", 
                       fontsize=8, verticalalignment='center')
            
            # Styling
            status = "✓ CORRECT" if correct else "✗ INCORRECT"
            ax.set_title(f"Query {i+1}: {query} - {status}", 
                        fontsize=11, fontweight='bold', loc='left')
            ax.set_ylabel('Rank', fontsize=10)
            ax.set_xlim(0, video_duration)
            ax.set_ylim(0, 6)
            ax.grid(True, alpha=0.3, axis='x')
            ax.legend(loc='upper right', fontsize=9)
            
            if i == n_queries - 1:
                ax.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Saved query comparison to {save_path}")
        else:
            default_path = self.output_dir / "query_comparison.png"
            plt.savefig(default_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Saved query comparison to {default_path}")
        
        plt.close()
    
    def export_pipeline_data(
        self,
        timestamps: List[float],
        raw_embeddings: np.ndarray,
        tas_outputs: Dict[str, np.ndarray],
        gru_output: np.ndarray,
        events: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ):
        """
        Export pipeline data to JSON for external analysis.
        
        Args:
            timestamps: Frame timestamps
            raw_embeddings: Raw embeddings
            tas_outputs: Dict with keys 'short', 'mid', 'long'
            gru_output: GRU output
            events: Detected events
            save_path: Path to save JSON
        """
        data = {
            'timestamps': timestamps,
            'pipeline': {
                'raw_shape': raw_embeddings.shape,
                'tas_short_shape': tas_outputs['short'].shape,
                'tas_mid_shape': tas_outputs['mid'].shape,
                'tas_long_shape': tas_outputs['long'].shape,
                'gru_shape': gru_output.shape
            },
            'events': events,
            'statistics': {
                'total_frames': len(timestamps),
                'total_events': len(events),
                'video_duration': max(timestamps) if timestamps else 0,
                'avg_frame_interval': np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0
            }
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Exported pipeline data to {save_path}")
        else:
            default_path = self.output_dir / "pipeline_data.json"
            with open(default_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Exported pipeline data to {default_path}")
