# Integration Tests for Cross-Modal Verification

## Overview

The integration tests in `test_context_aware_verification_integration.py` validate the complete pipeline of context-aware frame description with cross-modal verification.

## Test Structure

### Quick Tests (No Model Download Required)
- `test_synthetic_frame_creation`: Tests synthetic frame generation
- `test_video_sequence_creation`: Tests video sequence creation

These tests run quickly and don't require downloading models.

### Full Integration Tests (Requires Model Download)
- `test_context_buffer_management`: Tests FIFO context buffer behavior
- `test_context_used_tracking`: Tests context frame tracking
- `test_verification_basic`: Tests basic CLIP verification
- `test_verification_flagging`: Tests flagging of mismatched descriptions
- `test_end_to_end_pipeline`: Tests complete pipeline with 10 frames
- `test_entity_level_verification`: Tests entity-level verification

These tests require downloading:
- **CLIP ViT-B/32**: ~605MB (first run only)
- **SmolVLM-500M**: ~538MB (first run only)

Models are cached in `~/.cache/huggingface/` after first download.

## Running Tests

### Run All Tests (Including Model Download)
```bash
pytest tests/test_context_aware_verification_integration.py -v -s
```

**Note**: First run will take 5-10 minutes to download models (depending on internet speed).

### Run Quick Tests Only
```bash
pytest tests/test_context_aware_verification_integration.py -k "not Integration" -v
```

### Run Specific Test
```bash
pytest tests/test_context_aware_verification_integration.py::test_synthetic_frame_creation -v
```

### Run with Coverage
```bash
pytest tests/test_context_aware_verification_integration.py --cov=sharingan.verification --cov=sharingan.vlm -v
```

## Expected Output

### Successful Test Run
```
tests/test_context_aware_verification_integration.py::test_synthetic_frame_creation PASSED
tests/test_context_aware_verification_integration.py::test_video_sequence_creation PASSED
tests/test_context_aware_verification_integration.py::TestContextAwareVerificationIntegration::test_context_buffer_management PASSED
tests/test_context_aware_verification_integration.py::TestContextAwareVerificationIntegration::test_verification_basic PASSED
...
```

### Integration Test Output Example
```
================================================================================
INTEGRATION TEST: Context-Aware Description + Verification
================================================================================

✓ Frame 0 at 0.00s: VERIFIED
   Description: A gray image with text showing Frame 0.
   Alignment: 0.823

⚠️  Frame 1 at 0.50s: FLAGGED
   Description: A complex scene with multiple people dancing...
   Alignment: 0.542
   Flagged entities: ['people', 'dancing']
   Suggestion: Description may not match visual content (alignment: 0.542 < 0.700)...

================================================================================
SUMMARY: 10 frames processed
  Verified: 8
  Flagged: 2
================================================================================
```

## Requirements Validated

The integration tests validate the following requirements from the spec:

### Context-Aware Description (Requirements 2.x)
- **2.1**: SmolVLM maintains rolling context buffer of up to 8 frames
- **2.2**: Includes context from up to 8 previous frames in prompt
- **2.5**: Removes oldest frame when buffer exceeds 8 (FIFO)
- **2.6**: Records which context frames were used

### Cross-Modal Verification (Requirements 3.x)
- **3.1**: Computes CLIP similarity between frame and description
- **3.2**: Flags descriptions with CLIP similarity < 0.7
- **3.3**: Verifies each entity mentioned in description
- **3.4**: Flags entities with CLIP similarity < 0.5
- **3.5**: Provides correction suggestions when verification fails
- **3.6**: Returns VerificationResult with all required fields

## Troubleshooting

### Models Not Downloading
If models fail to download, check:
1. Internet connection
2. HuggingFace Hub access (no authentication required for public models)
3. Disk space (~2GB required for models)

### Out of Memory
If tests fail with OOM errors:
- Tests use `device="cpu"` by default to avoid GPU requirements
- Reduce `num_frames` in test sequences if needed
- Close other applications to free memory

### Slow Tests
First run is slow due to model download. Subsequent runs are much faster:
- First run: ~5-10 minutes (download + test)
- Subsequent runs: ~30-60 seconds (test only)

## CI/CD Integration

For CI/CD pipelines, consider:
1. Cache HuggingFace models between runs
2. Use `pytest -k "not Integration"` for quick checks
3. Run full integration tests only on main branch or releases
4. Set timeout to 15 minutes for first run with model download

Example GitHub Actions:
```yaml
- name: Run quick tests
  run: pytest tests/test_context_aware_verification_integration.py -k "not Integration" -v

- name: Run full integration tests (with model caching)
  run: pytest tests/test_context_aware_verification_integration.py -v
  timeout-minutes: 15
```
