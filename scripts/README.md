# SHARINGAN Scripts

This directory contains utility scripts for launching and managing SHARINGAN.

## Available Scripts

### `launch_ui.py`

Comprehensive launch script for the SHARINGAN Gradio web interface.

**Quick Start:**
```bash
python scripts/launch_ui.py
```

**Features:**
- Configurable host and port
- Public URL sharing via Gradio
- Optional authentication
- Pre-load processed videos from cache
- Comprehensive help text

**Common Usage:**

```bash
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

# Combine multiple options
python scripts/launch_ui.py --host 0.0.0.0 --port 8080 --share --auth admin:pass
```

**Full Documentation:**
See [docs/gradio_ui_launch.md](../docs/gradio_ui_launch.md) for complete documentation.

**Help:**
```bash
python scripts/launch_ui.py --help
```

## Script Development

When adding new scripts to this directory:

1. **Add shebang**: Start with `#!/usr/bin/env python3`
2. **Add docstring**: Include comprehensive module docstring
3. **Add help text**: Use argparse with detailed help messages
4. **Add examples**: Include usage examples in help text
5. **Handle errors**: Provide user-friendly error messages
6. **Update README**: Add script to this README

## Requirements

All scripts require the SHARINGAN package to be installed:

```bash
pip install -e .
```

Or install dependencies:

```bash
pip install -r requirements.txt
```

## Support

For issues or questions:
- **GitHub Issues**: https://github.com/yourusername/sharingan/issues
- **Documentation**: [docs/](../docs/)
