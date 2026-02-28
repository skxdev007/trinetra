# SHARINGAN UI Screenshots

This directory contains screenshots of the SHARINGAN Gradio UI for documentation purposes.

## Required Screenshots

The following screenshots are referenced in the main README.md:

1. **ui_upload.png** - Video upload and processing interface
   - Shows drag-and-drop upload area
   - Processing progress bar with ETA
   - Video preview panel

2. **ui_config.png** - Advanced configuration panel
   - Multi-Scale TAS settings
   - Adaptive sampler configuration
   - Model selection dropdowns
   - Preset profiles

3. **ui_query.png** - Query interface
   - Natural language input box
   - Example queries dropdown
   - Query history panel
   - Response display with timestamps

4. **ui_viz.png** - Visualizations and results
   - Causal graph network view
   - Event timeline
   - Reasoning scaffold display
   - Confidence score indicators

## How to Generate Screenshots

To generate these screenshots:

1. Launch the Gradio UI:
   ```bash
   python scripts/launch_ui.py
   ```

2. Upload a sample video and process it

3. Take screenshots of each section:
   - Use browser screenshot tools or OS screenshot utilities
   - Recommended resolution: 1920x1080 or higher
   - Save as PNG format for best quality

4. Name files according to the list above

5. Place files in this directory

## Placeholder Images

Until actual screenshots are available, the README will display placeholder text. The UI is fully functional and can be tested by running the launch script.

## Alternative: Remove Image References

If you prefer not to include screenshots, you can remove the image references from README.md:

```markdown
# Remove these lines:
![Video Upload Interface](docs/images/ui_upload.png)
![Configuration Panel](docs/images/ui_config.png)
![Query Interface](docs/images/ui_query.png)
![Visualizations](docs/images/ui_viz.png)
```

The documentation will still be complete and informative without the images.
