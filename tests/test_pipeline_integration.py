"""Integration test: run full pipeline with audio file source.

Requires:
- Ollama running with qwen2.5:7b-instruct pulled
- faster-whisper tiny.en model available

Mark as integration so unit tests remain fast.
"""
import json
import os
import pytest


@pytest.mark.integration
def test_full_pipeline_produces_output():
    """Launch pipeline with dev config and verify output file is produced."""
    # This test will run the Juturna pipeline programmatically
    # or via subprocess: python -m juturna launch -c streamind_dev.json
    # and check that results/ contains a window JSON file.
    pass  # Implementation depends on Juturna's programmatic API
