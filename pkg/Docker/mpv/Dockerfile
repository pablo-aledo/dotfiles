FROM ubuntu:zesty
LABEL maintainer "j"

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:mc3man/mpv-tests \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        mpv=2:0.27.0~zesty \
    && apt-get autoclean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

CMD [ "mpv" ]

