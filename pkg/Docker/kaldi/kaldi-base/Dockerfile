#######################################################################
# Dockerfile to build Kaldi (speech recognition engine container image
# Based on Ubuntu
#######################################################################

FROM debian:8

MAINTAINER Pawel Cyrta <pawel@cyrta.com>

################## BEGIN INSTALLATION ######################

#RUN sed -i "s/httpredir.debian.org/ftp.de.debian.org/" /etc/apt/sources.list

RUN apt-get clean

RUN apt-get update && apt-get install -y  \
    autoconf \
    automake \
    libtool-bin \
    make \
    gcc \
    g++ \
	gfortran \
    git \
    subversion \
    curl \
    wget \
    libjson0 \
    libjson0-dev \
    zlib1g-dev \
    bzip2 \
    gsl-bin libgsl0ldbl \
    libatlas3-base \
    glpk-utils


RUN apt-get update && apt-get install -y  \
    python2.7 \
    python-pip \
    python-yaml \
    python-simplejson \
    python-gi \
    python-software-properties && \
    pip install ws4py==0.3.2 && \
    pip install tornado && \
    ln -s /usr/bin/python2.7 /usr/bin/python ; ln -s -f bash /bin/sh

#------ Kaldi ----

RUN cd /opt && \
    echo "===> install Kaldi (latest from source)"  && \
    git clone https://github.com/kaldi-asr/kaldi && \
    cd /opt/kaldi/tools && \
    make && \
    ./install_portaudio.sh && \
    cd /opt/kaldi/src && ./configure --shared && \
    sed -i '/-g # -O0 -DKALDI_PARANOID/c\-O3 -DNDEBUG' kaldi.mk && \
    make depend && make && \
    cd /opt/kaldi/src/online && make depend && make && \
    cd /opt/kaldi/src/gst-plugin && make depend && make



#----- Cleaning up package manager
RUN apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*


ENTRYPOINT ["bin/bash"]
