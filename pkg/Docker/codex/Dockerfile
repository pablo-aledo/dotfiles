FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    docker.io \
    nodejs \
    npm \
    git \
    unzip \
    sudo \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @openai/codex --no-audit

RUN useradd -m -s /bin/bash codex && \
    usermod -aG docker codex && \
    echo "codex ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /agent-workspace

USER codex

CMD ["/bin/bash"]
