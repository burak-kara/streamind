#!/usr/bin/env python3
"""
Send an audio file to Janus as a WebRTC audio stream.

The script acts as a WebRTC publisher in Janus's VideoRoom plugin.  Once the
WebRTC connection is established, it calls rtp_forward to instruct Janus to
relay the audio as plain RTP to the pipeline's audio_rtp node.

Flow:
    audio file  →  aiortc (WebRTC)  →  Janus VideoRoom
                                             ↓  RTP/UDP
                                      pipeline audio_rtp node (:8888)

Any container format ffmpeg can decode is accepted (wav, opus, mp3, m4a, ...).
Duration is read via ffprobe so wav-only inspection is no longer required.

Usage:
    uv run python tools/send_audio.py datasets/rev16/10_Creating_Your_Own_Lane*.opus
    uv run python tools/send_audio.py audio.wav --pipeline-host 192.168.1.10
"""
import argparse
import asyncio
import logging
import subprocess
import time
import uuid

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _txn() -> str:
    return uuid.uuid4().hex[:12]


def audio_duration(path: str) -> float:
    """Return audio duration in seconds via ffprobe (format-agnostic)."""
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ])
    return float(out.strip())


async def _poll(client: httpx.AsyncClient, session_url: str) -> dict:
    """Long-poll Janus for the next non-keepalive event (~30s timeout)."""
    while True:
        rid = int(time.time() * 1000)
        resp = await client.get(f"{session_url}?rid={rid}", timeout=35.0)
        data = resp.json()
        if data.get("janus") != "keepalive":
            return data


# ── Main coroutine ────────────────────────────────────────────────────────────

async def send_audio(
    audio_path: str,
    janus_url: str = "http://localhost:8088/janus",
    room: int = 1234,
    pipeline_host: str = "host.docker.internal",
    pipeline_port: int = 8888,
) -> None:
    duration = audio_duration(audio_path)
    log.info("Audio: %s  (%.1fs)", audio_path, duration)

    async with httpx.AsyncClient() as client:

        # ── 1. Create Janus session ───────────────────────────────────────────
        r = await client.post(janus_url, json={"janus": "create", "transaction": _txn()})
        r.raise_for_status()
        session_id = r.json()["data"]["id"]
        session_url = f"{janus_url}/{session_id}"
        log.info("Session: %d", session_id)

        # ── 2. Attach to VideoRoom plugin ─────────────────────────────────────
        r = await client.post(session_url, json={
            "janus": "attach",
            "plugin": "janus.plugin.videoroom",
            "transaction": _txn(),
        })
        r.raise_for_status()
        handle_id = r.json()["data"]["id"]
        handle_url = f"{session_url}/{handle_id}"
        log.info("Handle: %d", handle_id)

        # ── 3. Join room as publisher ─────────────────────────────────────────
        await client.post(handle_url, json={
            "janus": "message",
            "transaction": _txn(),
            "body": {"request": "join", "room": room, "ptype": "publisher", "display": "streamind"},
        })
        event = await _poll(client, session_url)
        publisher_id = event["plugindata"]["data"]["id"]
        log.info("Joined room %d as publisher %d", room, publisher_id)

        # ── 4. Build peer connection with WAV audio track ─────────────────────
        pc = RTCPeerConnection()
        player = MediaPlayer(audio_path)
        pc.addTrack(player.audio)

        # Gather all ICE candidates before sending offer (non-trickle).
        gather_done = asyncio.Event()

        @pc.on("icegatheringstatechange")
        def _on_gathering():
            if pc.iceGatheringState == "complete":
                gather_done.set()

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        # Guard: event may fire before the handler is registered.
        if pc.iceGatheringState == "complete":
            gather_done.set()

        await asyncio.wait_for(gather_done.wait(), timeout=15.0)
        log.info("ICE gathering complete")

        # ── 5. Publish with SDP offer ─────────────────────────────────────────
        await client.post(handle_url, json={
            "janus": "message",
            "transaction": _txn(),
            "body": {"request": "publish", "audio": True, "video": False, "data": False},
            "jsep": {"type": "offer", "sdp": pc.localDescription.sdp},
        })
        event = await _poll(client, session_url)
        answer_sdp = event["jsep"]["sdp"]
        await pc.setRemoteDescription(RTCSessionDescription(type="answer", sdp=answer_sdp))
        log.info("SDP negotiation complete")

        # ── 6. Wait for ICE to connect ────────────────────────────────────────
        connected = asyncio.Event()

        @pc.on("connectionstatechange")
        async def _on_connection():
            log.info("Connection state: %s", pc.connectionState)
            if pc.connectionState == "connected":
                connected.set()
            elif pc.connectionState in ("failed", "closed"):
                connected.set()  # unblock; the state check below will raise

        await asyncio.wait_for(connected.wait(), timeout=30.0)

        if pc.connectionState != "connected":
            raise RuntimeError(f"WebRTC connection failed: state={pc.connectionState}")
        log.info("WebRTC connected")

        # ── 7. Instruct Janus to RTP-forward audio to the pipeline ───────────
        await client.post(handle_url, json={
            "janus": "message",
            "transaction": _txn(),
            "body": {
                "request": "rtp_forward",
                "room": room,
                "publisher_id": publisher_id,
                "host": pipeline_host,
                "audio_port": pipeline_port,
                "audio_pt": 97,
            },
        })
        event = await _poll(client, session_url)
        log.info("RTP forward active: %s", event.get("plugindata", {}).get("data", {}))

        # ── 8. Stream for the duration of the audio ──────────────────────────
        log.info("Streaming %.1fs of audio to pipeline...", duration)

        # Keep the Janus HTTP session alive by polling; without this the session
        # expires (~60s) and Janus tears down the WebRTC connection mid-stream.
        async def _keepalive_loop():
            while True:
                try:
                    rid = int(time.time() * 1000)
                    resp = await client.get(f"{session_url}?rid={rid}", timeout=35.0)
                    data = resp.json()
                    if data.get("janus") == "keepalive":
                        log.debug("Janus keepalive")
                    elif data.get("janus") != "ack":
                        log.debug("Janus event: %s", data.get("janus"))
                except Exception as e:
                    log.warning("Keepalive poll error: %s", e)
                    await asyncio.sleep(5)

        # Heartbeat: print elapsed/total every 60s so mid-stream truncation
        # is visible in stdout without log spelunking. Also surfaces WebRTC
        # connection state at each tick — if it flips off "connected", you
        # see the moment it happened.
        async def _heartbeat_loop():
            t_start = time.time()
            while True:
                await asyncio.sleep(60.0)
                elapsed = time.time() - t_start
                log.info(
                    "streamed %.0fs / %.0fs (%.0f%%)  pc.state=%s",
                    elapsed, duration, 100.0 * elapsed / max(duration, 1e-9),
                    pc.connectionState,
                )

        keepalive_task = asyncio.create_task(_keepalive_loop())
        heartbeat_task = asyncio.create_task(_heartbeat_loop())
        try:
            await asyncio.sleep(duration)
        finally:
            keepalive_task.cancel()
            heartbeat_task.cancel()
        log.info("streaming loop completed after %.1fs (target %.1fs)",
                 duration, duration)

        # ── 9. Tear down ─────────────────────────────────────────────────────
        await client.post(handle_url, json={
            "janus": "message", "transaction": _txn(),
            "body": {"request": "unpublish"},
        })
        await client.post(handle_url, json={"janus": "detach", "transaction": _txn()})
        await client.post(session_url, json={"janus": "destroy", "transaction": _txn()})
        await pc.close()
        log.info("Done")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream an audio file through Janus WebRTC to the STREAMIND pipeline."
    )
    parser.add_argument(
        "audio_path",
        help="Path to input audio file (any ffmpeg-decodable format: wav, opus, mp3, m4a, ...)",
    )
    parser.add_argument(
        "--janus-url", default="http://localhost:8088/janus",
        help="Janus HTTP API base URL (default: http://localhost:8088/janus)",
    )
    parser.add_argument(
        "--room", type=int, default=1234,
        help="VideoRoom room ID (default: 1234)",
    )
    parser.add_argument(
        "--pipeline-host", default="host.docker.internal",
        help="Host where the pipeline's audio_rtp node is listening (default: host.docker.internal)",
    )
    parser.add_argument(
        "--pipeline-port", type=int, default=8888,
        help="UDP port of the pipeline's audio_rtp node (default: 8888)",
    )
    args = parser.parse_args()

    asyncio.run(send_audio(
        audio_path=args.audio_path,
        janus_url=args.janus_url,
        room=args.room,
        pipeline_host=args.pipeline_host,
        pipeline_port=args.pipeline_port,
    ))


if __name__ == "__main__":
    main()
