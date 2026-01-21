FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    docker.io \
    sudo \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install aider-chat --break-system-packages

RUN useradd -m -s /bin/bash aider && \
    usermod -aG docker aider && \
    echo "aider ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /agent-workspace

USER aider

CMD ["/bin/bash"]
