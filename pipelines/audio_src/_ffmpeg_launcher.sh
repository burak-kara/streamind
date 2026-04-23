ffmpeg \
    -protocol_whitelist file,rtp,udp \
    -i /Users/burak/projects/streamind/pipelines/audio_src/_session_in.sdp \
    -f s16le \
    -ac 1 \
    -acodec pcm_s16le \
    -ar 16000 \
    -loglevel quiet \
    pipe:
