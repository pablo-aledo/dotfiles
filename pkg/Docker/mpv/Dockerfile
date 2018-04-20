FROM ubuntu

RUN apt-get update \
    && apt-get install -y mpv \
    && apt-get autoclean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT [ "mpv" ]

