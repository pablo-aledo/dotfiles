FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    docker.io \
    git \
    sudo \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs

WORKDIR /opt
RUN git clone https://github.com/badlogic/pi-mono.git

WORKDIR /opt/pi-mono
RUN npm install
RUN npm run build

RUN useradd -m -s /bin/bash pi && \
    usermod -aG docker pi && \
    echo "pi ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /agent-workspace

USER pi

ENV PATH="/opt/pi-mono/packages/coding-agent/bin:$PATH"

CMD ["/bin/bash"]
