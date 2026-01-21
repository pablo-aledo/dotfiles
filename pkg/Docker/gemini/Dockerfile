FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    docker.io \
    nodejs \
    npm \
    git \
    unzip \
    sudo \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @google/gemini-cli --no-audit

RUN useradd -m -s /bin/bash gemini && \
    usermod -aG docker gemini && \
    echo "gemini ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /agent-workspace

USER gemini

CMD ["/bin/bash"]
