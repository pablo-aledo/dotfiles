FROM       alpine

COPY repositories /etc/apk

RUN        apk update

RUN        apk add --no-cache -q \
               # tools
               alpine-sdk \
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
               curl-dev \
               dejagnu \
               flex \
               findutils \
               g++ \
               git \
               grep \
               #g++-multilib \
               jansson-dev \
               jemalloc-dev \
               make \
               libtool \
               llvm \
               m4 \
               #pkg-config \
               pkgconf \
               python \
               protobuf-dev \
               grpc-dev \
               strace \
               sudo \
               tmux \
               subversion \
               wget \
               yasm \
               zsh \
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
               sfml-dev \
               #libssl-dev \
               #libressl-dev \
               #openssl-dev \
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
               gsl-dev
               # ===== GL Visualization
               #libglfw-dev \
               #libglm-dev \
               #glm-dev \
               #glew-dev \
               #glfw-dev
               #libosmesa6-dev \
               #mesa-dev
               #libvtk5-dev \
               #libvtkedge-dev

RUN            mkdir /usr/src

RUN            ln -s /usr/lib/libgrpc++.so /usr/lib/libgrpc++.so.1                        && \
               ln -s /usr/lib/libgrpc++_reflection.so /usr/lib/libgrpc++_reflection.so.1

#RUN            curl -sSL https://github.com/uber/prototool/releases/download/v0.1.0/prototool-$(uname -s)-$(uname -m) \
                   #-o /usr/local/bin/prototool && \
                   #chmod +x /usr/local/bin/prototool

RUN            git clone https://github.com/oktal/pistache.git /usr/src/pistache && \
               cd /usr/src/pistache                                              && \
               git submodule update --init                                       && \
               mkdir build                                                       && \
               cd build                                                          && \
               cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..           && \
               make                                                              && \
               make install

RUN            git clone https://github.com/mrtazz/restclient-cpp.git /usr/src/restclient-cpp && \
               cd /usr/src/restclient-cpp                                                     && \
               ./autogen.sh                                                                   && \
               ./configure                                                                    && \
               make install

RUN            git clone https://github.com/jgaa/restc-cpp.git /usr/src/restc-cpp && \
               cd /usr/src/restc-cpp                                              && \
               mkdir build                                                        && \
               cd build                                                           && \
               cmake ..                                                           && \
               make                                                               && \
               make install
               
#RUN            git clone https://github.com/rizsotto/Bear.git /usr/src/bear && \
               #cd /usr/src/bear                                             && \
               #mkdir build && cd build                                      && \
               #cmake ..                                                     && \
               #make all                                                     && \
               #make install

#RUN            cd /usr/src                                                             && \
               #curl -L https://github.com/google/fruit/archive/v3.1.1.tar.gz | tar -xz && \
               #cd /usr/src/fruit-3.1.1                                                 && \
               #cmake -DCMAKE_BUILD_TYPE=Release -DFRUIT_USES_BOOST=False . && make -j  && \
               #make install                                                            && \
               #rm -rf /usr/src/fruit-3.1.1

#RUN            git clone -q https://github.com/google/googletest.git /usr/src/googletest && \
               #cd /usr/src/googletest                                                    && \
               #cmake .                                                                   && \
               #make                                                                      && \
               #make install                                                              && \
               #rm -rf /usr/src/googletest

#RUN            git clone https://github.com/google/protobuf.git /usr/src/protobuf && \
               #cd /usr/src/protobuf                                               && \
               #./autogen.sh                                                       && \
               #./configure --disable-shared                                       && \
               #make                                                               && \
               #make install                                                       && \
               #rm -rf /usr/src/protobuf

#RUN            git clone -b v1.8.x -q https://github.com/grpc/grpc /usr/src/grpc                    && \
               #cd /usr/src/grpc                                                                     && \
               #git submodule update --init                                                          && \
               #make                                                                                 && \
               #make install                                                                         && \
               #mv /usr/local/lib/libgrpc++.so.5 /usr/local/lib/libgrpc++.so.1                       && \
               #mv /usr/local/lib/libgrpc++_reflection.so.5 /usr/local/lib/libgrpc++_reflection.so.1 && \
               #rm -rf /usr/src/grpc

COPY       build.sh /usr/local/bin
RUN        chmod +x /usr/local/bin/build.sh

RUN        echo 'rmdir /root &  & ln -s /home/* /root' >> /bin/mounthome && \
           echo 'zsh' >> /bin/mounthome                                  && \
           chmod +x /bin/mounthome

RUN        echo 'wget tinyurl.com/pga-dfsi -O - | bash' >> /bin/pga && \
           echo 'zsh' >> /bin/pga                                   && \
           chmod +x /bin/pga

ENTRYPOINT     ["build.sh"]
