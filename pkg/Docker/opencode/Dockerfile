FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    docker.io \
    nodejs \
    npm \
    git \
    unzip \
    sudo \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g opencode-ai --no-audit

RUN useradd -m -s /bin/bash opencode && \
    usermod -aG docker opencode && \
    echo "opencode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /agent-workspace

USER opencode

CMD ["/bin/bash"]
