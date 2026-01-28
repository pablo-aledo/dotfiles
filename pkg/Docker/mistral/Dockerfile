FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    docker.io \
    curl \
    git \
    unzip \
    sudo \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash mistral && \
    usermod -aG docker mistral && \
    echo "mistral ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER mistral
WORKDIR /home/mistral

ENV PATH="/home/mistral/.local/bin:${PATH}"

RUN curl -LsSf https://mistral.ai/vibe/install.sh | bash

WORKDIR /agent-workspace

CMD ["/bin/bash"]
