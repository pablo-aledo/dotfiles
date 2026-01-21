FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    docker.io \
    git \
    unzip \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

RUN npm install -g @google/gemini-cli --no-audit

RUN useradd -m -s /bin/bash gemini && \
    usermod -aG docker gemini && \
    echo "gemini ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /agent-workspace

USER gemini

CMD ["/bin/bash"]
