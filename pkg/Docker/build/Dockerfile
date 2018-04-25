FROM       alpine

RUN        apk add --no-cache -q \
               # tools
               autoconf \
               automake \
               bash \
               bison \
               binutils \
               #build-essential \
               clang \
               cmake \
               cunit-dev \
               curl \
               dejagnu \
               flex \
               g++ \
               git \
               #g++-multilib \
               jansson-dev \
               jemalloc-dev \
               make \
               libtool           \
               llvm \
               m4 \
               #pkg-config \
               pkgconf \
               python \
               #protobuf-compiler \
               protobuf \
               protobuf-dev \
               protobuf-c-dev \
               subversion \
               wget \
               # useful libs
               c-ares-dev \
               #freeglut3-dev \
               #libboost-all-dev \
               boost-dev \
               #libc6-dev-i386 \
               libc6-compat \
               #libcrypto++-dev \
               #libfuse-dev \
               fuse-dev \
               #libgmp3-dev \
               #libgtk1.2-dev \
               #libgtk2.0-dev \
               gtk+2.0-dev \
               #libjpeg8-dev \
               jpeg-dev \
               #libmysql++-dev \
               libev-dev \
               libevent-dev \
               #mariadb-dev \
               #libncurses5-dev \
               ncurses-dev \
               #libreadline5-dev \
               readline-dev \
               #libsdl1.2-dev \
               #libsdl-image1.2-dev \
               #libsdl-mixer1.2-dev \
               sdl-dev \
               sdl_image-dev \
               sdl_mixer-dev \
               sdl2-dev \
               sdl2_image-dev \
               sdl2_mixer-dev \
               #libssl-dev \
               #libressl-dev \
               openssl-dev \
               libxi-dev \
               libxml2-dev \
               libxmu-dev \
               #libxmuu-dev \
               python-dev \
               #vflib3-dev \
               #xorg-dev \
               #zlib1g-dev \
               zlib-dev \
               # specific libs
               # ===== aircrack
               libpcap-dev \
               # ===== video
               #libavcodec52 \
               #libavcodec-extra-52 \
               #libavcodec-extra-53 \
               #libmp4v2-0 \
               #libxvidcore-dev \
               xvidcore-dev \
               # ===== vision
               #opencv-2 \
               #libcv2.1 \
               # ===== guis
               #libcairo2-dev \
               cairo-dev \
               #libqt4-dev \
               #poppler-qt4-dev \
               #libvtk5-qt4-dev \
               #libwxgtk2.8-dev \
               wxgtk2.8-dev \
               # ===== Scientific comp
               #libgsl0-dev \
               gsl-dev \
               # ===== GL Visualization
               #libglfw-dev \
               glfw-dev \
               #libglm-dev \
               glm-dev \
               #libosmesa6-dev \
               mesa-dev
               #libvtk5-dev \
               #libvtkedge-dev

RUN            mkdir /usr/src

RUN            cd /usr/src                                                             && \
               curl -L https://github.com/google/fruit/archive/v3.1.1.tar.gz | tar -xz && \
               cd /usr/src/fruit-3.1.1                                                 && \
               cmake -DCMAKE_BUILD_TYPE=Release -DFRUIT_USES_BOOST=False . && make -j  && \
               make install                                                            && \
               rm -rf /usr/src/fruit-3.1.1

RUN            git clone -q https://github.com/google/googletest.git /usr/src/googletest                     && \
               cd /usr/src/googletest                                                                        && \
               cmake .                                                                                       && \
               make                                                                                          && \
               make install                                                                                  && \
               rm -rf /usr/src/googletest

RUN            git clone  https://github.com/grpc/grpc /usr/src/grpc                                         && \
               cd /usr/src/grpc                                                                              && \
               git submodule update --init                                                                   && \
               make                                                                                          && \
               make install                                                                                  && \
               rm -rf /usr/src/grpc

RUN            cd /usr/src/                                                                                  && \
               curl -L https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-cpp-3.5.1.tar.gz    \
               | tar -xz                                                                                     && \
               cd /usr/src/protobuf-3.5.1                                                                    && \
               ./autogen.sh                                                                                  && \
               ./configure --disable-shared                                                                  && \
               make                                                                                          && \
               make install                                                                                  && \
               rm -rf /usr/src/protobuf-3.5.1

RUN            git clone https://github.com/rizsotto/Bear.git /usr/src/bear    && \
               cd /usr/src/bear                                                && \
               mkdir build && cd build                                         && \
               cmake ..                                                        && \
               make all                                                        && \
               make install                                                    && \
               rm -rf /usr/src/bear


COPY       build.sh /usr/local/bin
RUN        chmod +x /usr/local/bin/build.sh

ENTRYPOINT     build.sh
