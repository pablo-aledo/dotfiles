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

RUN useradd -m -s /bin/bash kimi && \
    usermod -aG docker kimi && \
    echo "kimi ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER kimi
WORKDIR /home/kimi

ENV PATH="/home/kimi/.local/bin:${PATH}"

RUN curl -L https://code.kimi.com/install.sh | bash

WORKDIR /agent-workspace

CMD ["/bin/bash"]
