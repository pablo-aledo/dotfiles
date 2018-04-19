FROM ubuntu:15.04
MAINTAINER Bradley Bossard <bradleybossard@gmail.com>

RUN apt-get update

RUN apt-get install -y git \
                       zip \
                       build-essential \
                       libboost-dev \
                       zlib1g-dev \
                       libpng12-dev \
                       libjpeg8-dev \
                       libtiff5-dev \
                       autoconf \
                       libopenexr-dev \
                       libboost-all-dev

RUN git clone https://github.com/POV-Ray/povray.git
WORKDIR /povray/unix
RUN ./prebuild.sh
WORKDIR /povray
RUN ./configure COMPILED_BY="Bradley Bossard <bradleybossard@gmail.com>"
RUN make
RUN make install
# TODO(bradleybossard) : Add a couple .pov sample files here to test that build worked.


# Removing this library.  It's required to build povray, but causes megapov build to fail.
RUN apt-get remove -y libpng12-dev

# This sets a default font, although it might not be the correct one
ENV SYSFONT latarcyrheb-sun16

WORKDIR /
ADD ./megapov-1.2.1.tgz megapov-1.2.1.tgz
WORKDIR /megapov-1.2.1.tgz/megapov-1.2.1/
RUN ./configure COMPILED_BY="bradleybossard@gmail.com"
RUN make
RUN make install

# TODO(bradleybossard) : Get these sample megapov file rendering.  Each has their own
# issues with either fonts or unable to find include files.
ADD ./attach.pov attach.pov
ADD ./bear.pov bear.pov
ADD ./clusglow.pov clusglow.pov
