# SHARINGAN Gradio UI Launch Guide

This guide explains how to launch and configure the SHARINGAN Gradio web interface using the `launch_ui.py` script.

## Quick Start

Launch with default settings (localhost only):

```bash
python scripts/launch_ui.py
```

This will start the UI at `http://localhost:7860`

## Command-Line Options

### Server Configuration

#### `--host HOST`
Specify the server host address.

- **Default**: `127.0.0.1` (localhost only)
- **Common values**:
  - `127.0.0.1` - Localhost only (most secure)
  - `0.0.0.0` - All network interfaces (allows remote access)
  - Specific IP address for your machine

**Examples:**
```bash
# Allow access from other devices on your network
python scripts/launch_ui.py --host 0.0.0.0

# Bind to specific IP
python scripts/launch_ui.py --host 192.168.1.100
```

#### `--port PORT`
Specify the server port number.

- **Default**: `7860`
- **Range**: 1024-65535 (avoid ports below 1024 which require admin privileges)

**Examples:**
```bash
# Use custom port
python scripts/launch_ui.py --port 8080

# Use port 80 (requires admin/sudo)
sudo python scripts/launch_ui.py --port 80
```

### Sharing Configuration

#### `--share`
Create a public URL for sharing your UI with others.

- Creates a temporary public URL (valid for 72 hours)
- Useful for demos, remote collaboration, or testing
- URL is provided by Gradio's sharing service

**Examples:**
```bash
# Create public URL
python scripts/launch_ui.py --share

# Combine with authentication for security
python scripts/launch_ui.py --share --auth admin:secretpass
```

**⚠️ Security Warning**: When using `--share`, your UI becomes publicly accessible. Always use `--auth` to protect it.

#### `--no-share`
Explicitly disable public URL sharing (default behavior).

```bash
python scripts/launch_ui.py --no-share
```

### Authentication Configuration

#### `--auth USERNAME:PASSWORD`
Enable password protection for the UI.

- Format: `username:password`
- Recommended when using `--share`
- Protects your UI from unauthorized access

**Examples:**
```bash
# Enable authentication
python scripts/launch_ui.py --auth admin:secretpass

# Combine with sharing
python scripts/launch_ui.py --share --auth demo:demo123
```

#### `--no-auth`
Explicitly disable authentication (default behavior).

```bash
python scripts/launch_ui.py --no-auth
```

### Cache Configuration

#### `--load-cache CACHE_DIR`
Load pre-processed videos from a cache directory.

- Allows resuming work on previously processed videos
- Saves time by not re-processing videos
- Cache directory must exist and contain valid cache files

**Examples:**
```bash
# Load from default cache directory
python scripts/launch_ui.py --load-cache ./cache

# Load from custom cache location
python scripts/launch_ui.py --load-cache /path/to/video/cache
```

**Cache File Format**: The cache directory should contain `.pkl` (pickle) files with processed video data.

### Advanced Options

#### `--debug`
Enable debug mode with verbose logging.

- Shows detailed error messages
- Displays stack traces for debugging
- Useful for troubleshooting issues

```bash
python scripts/launch_ui.py --debug
```

#### `--quiet`
Suppress non-error output.

- Minimal console output
- Only shows errors and warnings
- Useful for production deployments

```bash
python scripts/launch_ui.py --quiet
```

#### `--max-threads N`
Set maximum number of threads for the Gradio server.

- **Default**: Auto-detected based on CPU cores
- Higher values may improve concurrent request handling
- Lower values reduce memory usage

```bash
# Limit to 4 threads
python scripts/launch_ui.py --max-threads 4
```

## Common Usage Scenarios

### Local Development
```bash
# Simple local testing
python scripts/launch_ui.py

# With debug output
python scripts/launch_ui.py --debug
```

### Network Access
```bash
# Allow access from other devices on your network
python scripts/launch_ui.py --host 0.0.0.0 --port 8080

# With authentication for security
python scripts/launch_ui.py --host 0.0.0.0 --auth admin:pass123
```

### Public Demo
```bash
# Create public URL with authentication
python scripts/launch_ui.py --share --auth demo:demo123

# Load pre-processed demo videos
python scripts/launch_ui.py --share --auth demo:demo123 --load-cache ./demo_cache
```

### Production Deployment
```bash
# Production server with authentication
python scripts/launch_ui.py \
    --host 0.0.0.0 \
    --port 8080 \
    --auth admin:$(cat /etc/sharingan/password) \
    --load-cache /var/sharingan/cache \
    --quiet

# Behind nginx reverse proxy
python scripts/launch_ui.py \
    --host 127.0.0.1 \
    --port 7860 \
    --auth admin:secretpass \
    --quiet
```

### Resume Previous Work
```bash
# Load previously processed videos
python scripts/launch_ui.py --load-cache ./cache

# Continue work with specific configuration
python scripts/launch_ui.py \
    --load-cache ./cache \
    --host 0.0.0.0 \
    --port 8080
```

## Environment Variables

You can also configure the UI using environment variables:

```bash
# Set host and port via environment
export SHARINGAN_HOST=0.0.0.0
export SHARINGAN_PORT=8080
python scripts/launch_ui.py

# Set authentication via environment (more secure than command line)
export SHARINGAN_AUTH=admin:secretpass
python scripts/launch_ui.py --share
```

## Security Best Practices

### 1. Always Use Authentication with Public URLs
```bash
# ✅ GOOD: Public URL with authentication
python scripts/launch_ui.py --share --auth admin:strongpass

# ❌ BAD: Public URL without authentication
python scripts/launch_ui.py --share
```

### 2. Use Strong Passwords
```bash
# ✅ GOOD: Strong password
python scripts/launch_ui.py --auth admin:Xy9$mK2#pL5@

# ❌ BAD: Weak password
python scripts/launch_ui.py --auth admin:123456
```

### 3. Limit Network Access
```bash
# ✅ GOOD: Localhost only for development
python scripts/launch_ui.py --host 127.0.0.1

# ⚠️ CAUTION: All interfaces (use with authentication)
python scripts/launch_ui.py --host 0.0.0.0 --auth admin:pass
```

### 4. Use HTTPS in Production
For production deployments, use a reverse proxy (nginx, Apache) with HTTPS:

```nginx
# nginx configuration
server {
    listen 443 ssl;
    server_name sharingan.example.com;
    
    ssl_certificate /etc/ssl/certs/sharingan.crt;
    ssl_certificate_key /etc/ssl/private/sharingan.key;
    
    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Troubleshooting

### Port Already in Use
```bash
# Error: Address already in use
# Solution: Use a different port
python scripts/launch_ui.py --port 8080
```

### Permission Denied (Port < 1024)
```bash
# Error: Permission denied for port 80
# Solution: Use sudo or a higher port
sudo python scripts/launch_ui.py --port 80
# OR
python scripts/launch_ui.py --port 8080
```

### Cannot Access from Other Devices
```bash
# Problem: UI only accessible from localhost
# Solution: Bind to all interfaces
python scripts/launch_ui.py --host 0.0.0.0
```

### Cache Directory Not Found
```bash
# Error: Cache directory does not exist
# Solution: Create the directory first
mkdir -p ./cache
python scripts/launch_ui.py --load-cache ./cache
```

### Out of Memory
```bash
# Problem: Server crashes due to memory
# Solution: Reduce max threads
python scripts/launch_ui.py --max-threads 2
```

## Help and Documentation

### View All Options
```bash
python scripts/launch_ui.py --help
```

### View Version Information
```bash
python scripts/launch_ui.py --version
```

## System Requirements

### Minimum Requirements
- **RAM**: 8 GB
- **CPU**: 4 cores
- **GPU**: None (CPU-only mode, slower)
- **Disk**: 5 GB for models and cache

### Recommended Requirements
- **RAM**: 16 GB
- **CPU**: 8 cores
- **GPU**: NVIDIA GPU with 8 GB VRAM
- **Disk**: 20 GB for models, cache, and videos

### Optimal Requirements
- **RAM**: 32 GB
- **CPU**: 16 cores
- **GPU**: NVIDIA GPU with 16 GB VRAM
- **Disk**: 100 GB SSD for fast I/O

## Models Used

The UI automatically downloads and uses these models:

1. **SmolVLM-500M-Instruct** (~538 MB)
   - Vision-language model for frame descriptions
   - Provides context-aware video understanding

2. **CLIP ViT-B/32** (~400 MB)
   - Cross-modal verification
   - Detects VLM hallucinations

3. **Qwen2.5-0.5B-Instruct** (~538 MB)
   - Language model for query responses
   - Generates natural language answers

**Total Model Memory**: ~1.5 GB

## Performance Tips

### 1. Use GPU Acceleration
Ensure CUDA is installed and available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Adjust Batch Size
Modify configuration in the UI's advanced settings:
- Larger batch size = faster processing (requires more memory)
- Smaller batch size = slower processing (uses less memory)

### 3. Pre-process Videos
Process videos in advance and use `--load-cache`:
```bash
# Process videos offline
python scripts/process_videos.py --input ./videos --output ./cache

# Launch UI with pre-processed videos
python scripts/launch_ui.py --load-cache ./cache
```

### 4. Use Quantization
Enable 8-bit quantization in the UI settings to reduce memory usage by ~4x with minimal accuracy loss.

## Additional Resources

- **Main Documentation**: [README.md](../README.md)
- **Architecture Guide**: [ARCHITECTURE.md](../ARCHITECTURE.md)
- **UI Configuration**: [gradio_ui_configuration.md](./gradio_ui_configuration.md)
- **API Reference**: [api_reference.md](./api_reference.md)
- **Contributing**: [CONTRIBUTING.md](../CONTRIBUTING.md)

## Support

For issues, questions, or contributions:
- **GitHub Issues**: https://github.com/yourusername/sharingan/issues
- **Discussions**: https://github.com/yourusername/sharingan/discussions
- **Email**: support@sharingan.ai

## License

SHARINGAN is released under the MIT License. See [LICENSE](../LICENSE) for details.
