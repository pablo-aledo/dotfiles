FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    docker.io \
    git \
    sudo \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Instalar la última versión de Crush
RUN curl -L https://github.com/charmbracelet/crush/releases/download/v0.45.0/crush_0.45.0_Linux_x86_64.tar.gz \
    -o crush.tar.gz && \
    tar -xzf crush.tar.gz && \
    mv crush_0.45.0_Linux_x86_64/crush /usr/local/bin/crush && \
    chmod +x /usr/local/bin/crush && \
    rm crush.tar.gz

# Crear usuario no-root
RUN useradd -m -s /bin/bash crush && \
    usermod -aG docker crush && \
    echo "crush ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /agent-workspace

USER crush

CMD ["/bin/bash"]
