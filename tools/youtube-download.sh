#!/usr/bin/env bash
# Download a YouTube clip as mono 16kHz WAV for pipeline testing.
# Usage: ./tools/youtube-download.sh [URL] [duration]
#   defaults: first 20 min of the hardcoded URL below

set -euo pipefail

URL="${1:-https://www.youtube.com/watch?v=lBVtvOpU80Q}"
DURATION="${2:-00:20:00}"
OUT_DIR="./tmp"
mkdir -p "$OUT_DIR"

yt-dlp \
  --download-sections "*00:00:00-${DURATION}" \
  -f bestaudio \
  -x --audio-format wav --audio-quality 0 \
  -o "${OUT_DIR}/youtube_raw.%(ext)s" \
  "$URL"

ffmpeg -y \
  -i "${OUT_DIR}/youtube_raw.wav" \
  -ac 1 -ar 16000 -c:a pcm_s16le \
  "${OUT_DIR}/youtube_audio.wav"

echo "Output: ${OUT_DIR}/youtube_audio.wav"
