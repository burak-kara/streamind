brew install yt-dlp ffmpeg

yt-dlp \
  --download-sections "*00:00:00-00:20:00" \
  -f bestaudio \
  -x --audio-format wav --audio-quality 0 \
  -o "tests/fixtures/youtube_raw.%(ext)s" \
  "https://www.youtube.com/watch?v=lBVtvOpU80Q"

ffmpeg -y \
  -i tests/fixtures/youtube_raw.wav \
  -ac 1 -ar 16000 -c:a pcm_s16le \
  tests/fixtures/youtube_15min.wav