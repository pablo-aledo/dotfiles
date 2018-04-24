FROM ubuntu:14.04

RUN apt-get update -y && apt-get install -y \
    clang \
    binutils \
    wget \
    libxi6 \
    libxrender1 \
    make \
    libxtst6 && \
    mkdir ~/clion && \
    cd ~/clion && \
    wget https://download.jetbrains.com/cpp/clion-1.2.4.tar.gz -O - | tar -xz

RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang 100 

CMD /root/clion/clion-1.2.4/bin/clion.sh

