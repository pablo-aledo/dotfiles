FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    docker.io \
    nodejs \
    npm \
    git \
    unzip \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code --no-audit

RUN useradd -m -s /bin/bash claude && \
    usermod -aG docker claude && \
    echo "claude ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /agent-workspace

USER claude

CMD ["/bin/bash"]

