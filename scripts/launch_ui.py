#!/usr/bin/env python3
"""
============================================================================
SHARINGAN Gradio UI Launch Script
============================================================================

This script provides a command-line interface for launching the SHARINGAN
Gradio web UI with various configuration options.

USAGE:
    python scripts/launch_ui.py [OPTIONS]

EXAMPLES:
    # Launch with default settings (localhost:7860)
    python scripts/launch_ui.py

    # Launch on custom host and port
    python scripts/launch_ui.py --host 0.0.0.0 --port 8080

    # Create public URL for sharing
    python scripts/launch_ui.py --share

    # Enable authentication
    python scripts/launch_ui.py --auth username:password

    # Load pre-processed videos from cache
    python scripts/launch_ui.py --load-cache ./cache

    # Disable authentication (default)
    python scripts/launch_ui.py --no-auth

FEATURES:
    - Configurable host and port
    - Public URL sharing via Gradio
    - Optional authentication
    - Pre-load processed videos from cache
    - Comprehensive help text

============================================================================
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple


def parse_arguments():
    """
    Parse command-line arguments for launching the Gradio UI.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Launch SHARINGAN Gradio Web UI for video understanding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  Launch with default settings:
    python scripts/launch_ui.py

  Launch on all network interfaces:
    python scripts/launch_ui.py --host 0.0.0.0

  Launch with custom port:
    python scripts/launch_ui.py --port 8080

  Create public URL for sharing:
    python scripts/launch_ui.py --share

  Enable authentication:
    python scripts/launch_ui.py --auth admin:secretpass

  Load pre-processed videos:
    python scripts/launch_ui.py --load-cache ./cache

  Combine multiple options:
    python scripts/launch_ui.py --host 0.0.0.0 --port 8080 --share --auth user:pass

ABOUT SHARINGAN:
  SHARINGAN is a deep video understanding system that processes videos once
  and enables unlimited querying with zero API cost. It uses multi-scale
  temporal reasoning, context-aware VLMs, and causal graph construction.

  Key Features:
  - Process video once, query forever (no re-processing)
  - Fast queries (<500ms) with local models
  - Complete privacy (no external API calls)
  - Temporal reasoning (understands "why" and "when")
  - Zero API cost (runs locally)

MODELS:
  - SmolVLM-500M: Vision-language model for frame descriptions
  - CLIP ViT-B/32: Cross-modal verification
  - Qwen2.5-0.5B: Language model for query responses
  
  Total model memory: ~1.5 GB

HARDWARE REQUIREMENTS:
  Minimum: 8 GB RAM, CPU-only (slow)
  Recommended: 16 GB RAM, NVIDIA GPU with 8 GB VRAM
  Optimal: 32 GB RAM, NVIDIA GPU with 16 GB VRAM

For more information, visit: https://github.com/yourusername/sharingan
        """
    )
    
    # Server configuration
    server_group = parser.add_argument_group('Server Configuration')
    server_group.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Server host address (default: 127.0.0.1 for localhost only). '
             'Use 0.0.0.0 to allow access from other devices on the network.'
    )
    server_group.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Server port number (default: 7860)'
    )
    
    # Sharing configuration
    sharing_group = parser.add_argument_group('Sharing Configuration')
    sharing_group.add_argument(
        '--share',
        action='store_true',
        help='Create a public URL for sharing (valid for 72 hours). '
             'Useful for demos and remote access.'
    )
    sharing_group.add_argument(
        '--no-share',
        dest='share',
        action='store_false',
        help='Disable public URL sharing (default)'
    )
    parser.set_defaults(share=False)
    
    # Authentication configuration
    auth_group = parser.add_argument_group('Authentication Configuration')
    auth_group.add_argument(
        '--auth',
        type=str,
        default=None,
        metavar='USERNAME:PASSWORD',
        help='Enable authentication with username:password format. '
             'Example: --auth admin:secretpass'
    )
    auth_group.add_argument(
        '--no-auth',
        dest='auth',
        action='store_const',
        const=None,
        help='Disable authentication (default)'
    )
    
    # Cache configuration
    cache_group = parser.add_argument_group('Cache Configuration')
    cache_group.add_argument(
        '--load-cache',
        type=str,
        default=None,
        metavar='CACHE_DIR',
        help='Load pre-processed videos from cache directory. '
             'This allows you to resume work on previously processed videos. '
             'Example: --load-cache ./cache'
    )
    
    # Advanced options
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    advanced_group.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress non-error output'
    )
    advanced_group.add_argument(
        '--max-threads',
        type=int,
        default=None,
        help='Maximum number of threads for Gradio server (default: auto)'
    )
    
    return parser.parse_args()


def parse_auth(auth_string: Optional[str]) -> Optional[Tuple[str, str]]:
    """
    Parse authentication string in format 'username:password'.
    
    Args:
        auth_string: Authentication string or None
    
    Returns:
        Tuple of (username, password) or None if auth is disabled
    
    Raises:
        ValueError: If auth string format is invalid
    """
    if auth_string is None:
        return None
    
    if ':' not in auth_string:
        raise ValueError(
            "Invalid authentication format. Use 'username:password' format.\n"
            "Example: --auth admin:secretpass"
        )
    
    parts = auth_string.split(':', 1)
    username, password = parts[0], parts[1]
    
    if not username or not password:
        raise ValueError(
            "Username and password cannot be empty.\n"
            "Example: --auth admin:secretpass"
        )
    
    return (username, password)


def validate_cache_directory(cache_dir: Optional[str]) -> Optional[Path]:
    """
    Validate cache directory path.
    
    Args:
        cache_dir: Cache directory path or None
    
    Returns:
        Path object or None if cache is not used
    
    Raises:
        ValueError: If cache directory is invalid
    """
    if cache_dir is None:
        return None
    
    path = Path(cache_dir)
    
    if not path.exists():
        raise ValueError(
            f"Cache directory does not exist: {cache_dir}\n"
            f"Please create the directory or provide a valid path."
        )
    
    if not path.is_dir():
        raise ValueError(
            f"Cache path is not a directory: {cache_dir}\n"
            f"Please provide a valid directory path."
        )
    
    return path


def load_cached_videos(cache_dir: Path) -> int:
    """
    Load pre-processed videos from cache directory.
    
    Args:
        cache_dir: Path to cache directory
    
    Returns:
        Number of videos loaded
    """
    # Import here to avoid circular imports
    from sharingan.ui.gradio_app import PROCESSED_VIDEOS
    
    count = 0
    
    # Look for .pkl or .json files in cache directory
    for cache_file in cache_dir.glob('*.pkl'):
        try:
            # Load cached video data
            import pickle
            with open(cache_file, 'rb') as f:
                video_data = pickle.load(f)
            
            # Add to global state
            video_key = cache_file.stem
            PROCESSED_VIDEOS[video_key] = video_data
            count += 1
            
            print(f"✓ Loaded cached video: {video_key}")
            
        except Exception as e:
            print(f"✗ Failed to load {cache_file.name}: {e}")
    
    return count


def print_banner():
    """Print SHARINGAN banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ███████╗██╗  ██╗ █████╗ ██████╗ ██╗███╗   ██╗ ██████╗ █████╗ ███╗   ██╗   ║
    ║   ██╔════╝██║  ██║██╔══██╗██╔══██╗██║████╗  ██║██╔════╝██╔══██╗████╗  ██║   ║
    ║   ███████╗███████║███████║██████╔╝██║██╔██╗ ██║██║  ███╗███████║██╔██╗ ██║   ║
    ║   ╚════██║██╔══██║██╔══██║██╔══██╗██║██║╚██╗██║██║   ██║██╔══██║██║╚██╗██║   ║
    ║   ███████║██║  ██║██║  ██║██║  ██║██║██║ ╚████║╚██████╔╝██║  ██║██║ ╚████║   ║
    ║   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ║
    ║                                                               ║
    ║              Deep Video Understanding System                  ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main entry point for launch script."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Print banner unless quiet mode
        if not args.quiet:
            print_banner()
            print("\n🚀 Launching SHARINGAN Gradio UI...\n")
        
        # Validate and parse authentication
        auth = None
        if args.auth:
            try:
                auth = parse_auth(args.auth)
                if not args.quiet:
                    print(f"🔒 Authentication enabled (user: {auth[0]})")
            except ValueError as e:
                print(f"❌ Error: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            if not args.quiet:
                print("🔓 Authentication disabled")
        
        # Validate and load cache
        cache_dir = None
        if args.load_cache:
            try:
                cache_dir = validate_cache_directory(args.load_cache)
                if not args.quiet:
                    print(f"📂 Loading cached videos from: {cache_dir}")
                
                count = load_cached_videos(cache_dir)
                if not args.quiet:
                    print(f"✓ Loaded {count} cached video(s)\n")
                    
            except ValueError as e:
                print(f"❌ Error: {e}", file=sys.stderr)
                sys.exit(1)
        
        # Print configuration
        if not args.quiet:
            print("⚙️  Configuration:")
            print(f"   Host: {args.host}")
            print(f"   Port: {args.port}")
            print(f"   Share: {'Yes (public URL will be created)' if args.share else 'No'}")
            print(f"   Auth: {'Enabled' if auth else 'Disabled'}")
            if cache_dir:
                print(f"   Cache: {cache_dir}")
            if args.max_threads:
                print(f"   Max Threads: {args.max_threads}")
            print()
        
        # Import and launch Gradio app
        from sharingan.ui.gradio_app import launch_app
        
        if not args.quiet:
            print("🌐 Starting server...")
            print(f"📍 Local URL: http://{args.host}:{args.port}")
            if args.share:
                print("🔗 Public URL will be displayed below...")
            print("\n" + "="*60)
            print("Press Ctrl+C to stop the server")
            print("="*60 + "\n")
        
        # Launch with configuration
        launch_app(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            auth=auth
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down SHARINGAN UI...")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
