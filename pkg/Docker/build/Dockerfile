FROM       alpine

COPY repositories /etc/apk

RUN        apk update

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
	       protobuf-dev \
	       grpc-dev \
               strace \
               subversion \
               wget \
               yasm \
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
               libxcursor-dev \
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
               libxrandr-dev \
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
               ffmpeg-dev \
               xvidcore-dev \
               x264-dev \
               libxinerama-dev \
               # ===== vision
               #opencv-2 \
	       opencv-dev \
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
               glew-dev \
               glfw-dev \
               #libosmesa6-dev \
               mesa-dev
               #libvtk5-dev \
               #libvtkedge-dev

RUN            mkdir /usr/src

RUN            ln -s /usr/lib/libgrpc++.so /usr/lib/libgrpc++.so.1                        && \
               ln -s /usr/lib/libgrpc++_reflection.so /usr/lib/libgrpc++_reflection.so.1

RUN            git clone https://github.com/rizsotto/Bear.git /usr/src/bear               && \
               cd /usr/src/bear                                                           && \
               mkdir build && cd build                                                    && \
               cmake ..                                                                   && \
               make all                                                                   && \
               make install                                                               && \
               rm -rf /usr/src/bear

COPY       build.sh /usr/local/bin
RUN        chmod +x /usr/local/bin/build.sh

ENTRYPOINT     ["build.sh"]

