FROM ubuntu:16.04

LABEL maintainer "Darren Green <darren@gruen.site>"

VOLUME ["${HOME}/Downloads", "${HOME}/Documents", "${HOME}/.dia"]

ENTRYPOINT ["dia"]

RUN apt-get update \
    && apt-get install -y dia \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

