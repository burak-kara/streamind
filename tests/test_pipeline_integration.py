"""Integration test: run full pipeline through Janus WebRTC.

Requires:
- Janus running:  docker compose up -d
- Ollama running: ollama serve  (with qwen3.5:9b-16k pulled)
- faster-whisper small.en model (downloads automatically on first run)

Run with:  .venv/bin/pytest tests/test_pipeline_integration.py -m integration -v
Skipped by default in the unit-test suite: .venv/bin/pytest tests/
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SAMPLE_WAV = REPO_ROOT / "tests" / "fixtures" / "sample_audio_15min.wav"
RESULTS_DIR = REPO_ROOT / "results"
SEND_AUDIO = REPO_ROOT / "tools" / "send_audio.py"


def _janus_available() -> bool:
    try:
        import httpx
        r = httpx.get("http://localhost:8088/janus/info", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _ollama_available() -> bool:
    try:
        import httpx
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(autouse=True)
def clean_results():
    """Remove stale result files before and after each test."""
    for f in RESULTS_DIR.glob("window_*.json"):
        f.unlink(missing_ok=True)
    yield
    for f in RESULTS_DIR.glob("window_*.json"):
        f.unlink(missing_ok=True)


@pytest.fixture
def pipeline_proc():
    """Launch the production pipeline; terminate on teardown."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "juturna", "launch", "--config",
         str(REPO_ROOT / "pipelines" / "config.json")],
        cwd=str(REPO_ROOT),
    )
    # Allow time for Whisper model to load before audio arrives.
    time.sleep(5)
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.mark.integration
def test_full_pipeline_with_janus(pipeline_proc):
    """End-to-end: WAV → Janus → audio_rtp → pipeline → results/window_N.json.

    Uses a 10-minute fixture so the 300s production window fires at least once
    during normal operation without requiring any config changes.
    """
    if not SAMPLE_WAV.exists():
        pytest.skip(f"Test fixture not found: {SAMPLE_WAV}")
    if not _janus_available():
        pytest.skip("Janus not running — start with: docker compose up -d")
    if not _ollama_available():
        pytest.skip("Ollama not running on localhost:11434")

    # Stream the WAV through Janus; blocks until the full file is sent (~650s).
    subprocess.run(
        [sys.executable, str(SEND_AUDIO), str(SAMPLE_WAV)],
        check=True,
        timeout=900,
    )

    # The 300s window fires at ~300s into the audio; wait up to 60s after audio
    # ends for the LLM to finish processing the final window.
    deadline = time.time() + 60
    output_files: list[Path] = []
    while time.time() < deadline:
        output_files = sorted(RESULTS_DIR.glob("window_*.json"))
        if output_files:
            break
        time.sleep(2)

    assert output_files, "No output files produced within 60s after audio finished"

    for filepath in output_files:
        data = json.loads(filepath.read_text())

        assert "window_id" in data, \
            f"{filepath.name}: missing window_id"
        assert isinstance(data["summary"], str) and data["summary"].strip(), \
            f"{filepath.name}: summary is empty"
        assert isinstance(data["keywords"], list), \
            f"{filepath.name}: keywords not a list"
        assert len(data["keywords"]) == 3, \
            f"{filepath.name}: expected 3 keywords, got {len(data['keywords'])}"
        assert all(isinstance(k, str) for k in data["keywords"]), \
            f"{filepath.name}: all keywords must be strings"
        assert isinstance(data["latency"], float) and data["latency"] > 0, \
            f"{filepath.name}: latency must be a positive float"
        assert data["latency"] < 30.0, \
            f"{filepath.name}: latency {data['latency']:.1f}s exceeds 30s target"
