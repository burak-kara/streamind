"""Integration test: run the full vLLM pipeline through Janus WebRTC.

Requires uni-lab (RTX 4090, CUDA 12.4) and:
- `uv sync --extra dev`
- Janus running:  `docker compose up -d`
- Model weights present under `./models/<name>/` (run
  `./tools/fetch_models.sh <hf_id> <name>` first)
- faster-whisper small.en (downloads automatically on first run)

Run with:  `uv run pytest tests/test_pipeline_integration.py -m integration -v`
Skipped by default in the unit-test suite: `uv run pytest tests/`
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


def _vllm_importable() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except Exception:
        return False


def _active_summarizer_profile() -> Path | None:
    """Pick the first vllm-*.json profile, mirroring the dev workflow."""
    candidates = sorted((REPO_ROOT / "pipelines" / "summarizer").glob("vllm-*.json"))
    return candidates[0] if candidates else None


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
    """Launch the assembled pipeline via run_pipeline.sh; terminate on teardown."""
    profile = _active_summarizer_profile()
    if profile is None:
        pytest.skip("No pipelines/summarizer/vllm-*.json profile committed yet")

    proc = subprocess.Popen(
        ["bash", str(REPO_ROOT / "tools" / "run_pipeline.sh"),
         "--summarizer", profile.stem],
        cwd=str(REPO_ROOT),
    )
    # Allow time for vLLM warmup + Whisper model load before audio arrives.
    time.sleep(30)
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.mark.integration
def test_full_pipeline_with_janus(pipeline_proc):
    """End-to-end: WAV → Janus → audio_rtp → pipeline → results/window_N.json."""
    if not SAMPLE_WAV.exists():
        pytest.skip(f"Test fixture not found: {SAMPLE_WAV}")
    if not _janus_available():
        pytest.skip("Janus not running — start with: docker compose up -d")
    if not _vllm_importable():
        pytest.skip("vLLM not importable — run `uv sync` on a CUDA host (uni-lab)")

    profile = _active_summarizer_profile()
    cfg = json.loads(profile.read_text())
    model_path = Path(cfg["configuration"]["model_name"])
    if not (model_path / "config.json").exists():
        pytest.skip(f"Model weights missing at {model_path} — run tools/fetch_models.sh")

    # Stream the WAV through Janus; blocks until the full file is sent.
    subprocess.run(
        [sys.executable, str(SEND_AUDIO), str(SAMPLE_WAV)],
        check=True,
        timeout=900,
    )

    deadline = time.time() + 60
    output_files: list[Path] = []
    while time.time() < deadline:
        output_files = sorted(RESULTS_DIR.glob("**/window_*.json"))
        output_files = [p for p in output_files if "judge" not in p.parts and "debug" not in p.parts]
        if output_files:
            break
        time.sleep(2)

    assert output_files, "No output files produced within 60s after audio finished"

    for filepath in output_files:
        data = json.loads(filepath.read_text())
        # Challenge output contract: from, to, summary, keywords (3), proc_time.
        assert "from" in data and "to" in data, f"{filepath.name}: missing from/to"
        assert isinstance(data["summary"], str) and data["summary"].strip(), \
            f"{filepath.name}: summary is empty"
        assert isinstance(data["keywords"], list) and len(data["keywords"]) == 3, \
            f"{filepath.name}: expected 3 keywords, got {data['keywords']}"
        assert all(isinstance(k, str) for k in data["keywords"]), \
            f"{filepath.name}: all keywords must be strings"
        assert isinstance(data["proc_time"], (int, float)) and data["proc_time"] > 0, \
            f"{filepath.name}: proc_time must be a positive number"
