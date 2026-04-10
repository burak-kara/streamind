# Janus WebRTC Gateway — built from source following the official README.
# Uses the 0.x (stable legacy) branch for a well-documented, simple HTTP API.
#
# First build takes ~15 minutes; subsequent builds are cached.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── Build dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential autoconf automake libtool pkg-config git gengetopt \
    libmicrohttpd-dev libjansson-dev libssl-dev \
    libsofia-sip-ua-dev libglib2.0-dev \
    libopus-dev libogg-dev libcurl4-openssl-dev \
    libconfig-dev \
    python3 python3-pip meson ninja-build \
    && rm -rf /var/lib/apt/lists/*

# ── libnice from source ───────────────────────────────────────────────────────
# Ubuntu's packaged libnice is known to cause ICE problems (per Janus README).
RUN git clone --depth 1 https://gitlab.freedesktop.org/libnice/libnice /tmp/libnice \
    && cd /tmp/libnice \
    && meson --prefix=/usr build \
    && ninja -C build \
    && ninja -C build install \
    && rm -rf /tmp/libnice

# ── libsrtp 2.x from source ──────────────────────────────────────────────────
RUN git clone --depth 1 --branch v2.6.0 https://github.com/cisco/libsrtp /tmp/libsrtp \
    && cd /tmp/libsrtp \
    && ./configure --prefix=/usr --enable-openssl \
    && make shared_library \
    && make install \
    && rm -rf /tmp/libsrtp

# ── Janus 0.x (stable legacy — simple single-stream VideoRoom API) ────────────
RUN git clone --depth 1 --branch 0.x https://github.com/meetecho/janus-gateway.git /tmp/janus \
    && cd /tmp/janus \
    && sh autogen.sh \
    && ./configure \
        --prefix=/opt/janus \
        --disable-websockets \
        --disable-data-channels \
        --disable-rabbitmq \
        --disable-mqtt \
    && make -j"$(nproc)" \
    && make install \
    && make configs \
    && rm -rf /tmp/janus

# ── Project-specific config (overrides make configs defaults) ─────────────────
COPY conf/ /opt/janus/etc/janus/

EXPOSE 8088
EXPOSE 10000-10050/udp

CMD ["/opt/janus/bin/janus"]
