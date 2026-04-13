# yt-dlp Quick Reference

## Installation

```bash
brew install yt-dlp
brew install ffmpeg  # required for audio extraction and conversion
```

## Audio Extraction

```bash
# Best quality audio as mp3
yt-dlp -x --audio-format mp3 "URL"

# Specify output path and filename
yt-dlp -x --audio-format mp3 -o "~/Downloads/%(title)s.%(ext)s" "URL"

# Best quality flag (0 = best, 9 = worst)
yt-dlp -x --audio-format mp3 --audio-quality 0 "URL"
```

## Supported Audio Formats

`mp3` · `m4a` · `opus` · `flac` · `wav` · `aac`

## WAV Extraction

```bash
# Download and convert directly to WAV (PCM, lossless)
yt-dlp -x --audio-format wav "URL"

# Keep original alongside the converted WAV
yt-dlp -x --audio-format wav -k "URL"
```

> WAV output is uncompressed PCM. File sizes will be large — expect ~10 MB/min for stereo 44.1 kHz 16-bit.

## Opus to WAV Conversion (via ffmpeg)

If you already have an `.opus` file and need to convert it:

```bash
ffmpeg -i input.opus -c:a pcm_s16le output.wav
```

Common PCM format options:

| Flag | Format |
|---|---|
| `pcm_s16le` | 16-bit signed, little-endian (standard) |
| `pcm_s24le` | 24-bit signed, little-endian (higher depth) |
| `pcm_f32le` | 32-bit float (DAW-friendly) |

To also resample (e.g., to 44100 Hz):

```bash
ffmpeg -i input.opus -c:a pcm_s16le -ar 44100 output.wav
```

## Useful Flags

| Flag | Description |
|---|---|
| `-x` | Extract audio only |
| `--audio-format FORMAT` | Convert to target format |
| `--audio-quality 0` | Set quality (0 = best) |
| `-o "TEMPLATE"` | Output filename template |
| `--no-playlist` | Download single video, not full playlist |
| `-k` | Keep original file after conversion |

## Updates

```bash
brew upgrade yt-dlp
```

## Notes

- ffmpeg must be installed and on PATH for format conversion to work.
- Downloading copyrighted content may violate platform Terms of Service.
