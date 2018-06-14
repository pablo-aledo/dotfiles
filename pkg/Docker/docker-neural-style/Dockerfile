FROM ubuntu:14.04

# Based on:
# https://github.com/Kaixhin/dockerfiles/blob/master/torch/Dockerfile
# https://github.com/kvvzr/docker-neural-style/blob/master/Dockerfile

RUN apt-get update && apt-get install -y \
  curl \
  ipython3 \
  libpng-dev \
  libprotobuf-dev \
  protobuf-compiler \
  python-zmq \
  wget

# Torch7
RUN curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
RUN git clone https://github.com/torch/distro.git ~/torch --recursive && \
  (cd ~/torch; ./install.sh)

# neural-style
RUN /root/torch/install/bin/luarocks install loadcaffe

# Export environment variables manually
ENV LUA_PATH='/neural-style/?.lua;/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-alpha/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua' \
  LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so' \
  PATH=/root/torch/install/bin:$PATH \
  LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH \
  DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH

WORKDIR /neural-style
VOLUME /neural-style

ENTRYPOINT ["/root/torch/install/bin/th", "/neural-style/neural_style.lua"]
