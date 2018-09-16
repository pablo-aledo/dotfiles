
FROM cyrta/kaldi-base

MAINTAINER Pawel Cyrta <pawel@cyrta.com>


################## BEGIN INSTALLATION ######################


RUN apt-get clean

#to have ffmpeg
RUN echo "deb http://www.deb-multimedia.org jessie main non-free" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --force-yes deb-multimedia-keyring

#--- dependencies ---

RUN apt-get update && apt-get install -y  \
	supervisor
    sox \
    libsox-fmt-* \
    lame \
    libmp3lame-dev \
    ffmpeg \
    libavcodec56 libavformat56 libavfilter5 libavdevice56 \
    gstreamer1.0-plugins-good \
    gstreamer1.0-tools \
    gstreamer1.0-pulseaudio \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly  \
    libgstreamer1.0-dev

RUN cd /opt && wget http://www.digip.org/jansson/releases/jansson-2.7.tar.bz2 && \
    bunzip2 -c jansson-2.7.tar.bz2 | tar xf -  && \
    cd jansson-2.7 && \
    ./configure && make && make check &&  make install && \
    echo "/usr/local/lib" >> /etc/ld.so.conf.d/jansson.conf && ldconfig && \
    rm /opt/jansson-2.7.tar.bz2 && rm -rf /opt/jansson-2.7


#------ Kaldi GStreamer ----

RUN cd /opt && \
    echo "===> install Kaldi GStreamer Plugin"  && \
    git clone https://github.com/alumae/gst-kaldi-nnet2-online.git && \
    cd /opt/gst-kaldi-nnet2-online/src && \
    sed -i '/KALDI_ROOT?=\/home\/tanel\/tools\/kaldi-trunk/c\KALDI_ROOT?=\/opt\/kaldi' Makefile && \
    make depend && make && \
    rm -rf /opt/gst-kaldi-nnet2-online/.git/ && \
    find /opt/gst-kaldi-nnet2-online/src/ -type f -not -name '*.so' -delete && \
    rm -rf /opt/kaldi/.git && \
    rm -rf /opt/kaldi/egs/ /opt/kaldi/windows/ /opt/kaldi/misc/ && \
    find /opt/kaldi/src/ -type f -not -name '*.so' -delete && \
    find /opt/kaldi/tools/ -type f \( -not -name '*.so' -and -not -name '*.so*' \) -delete && \

RUN cd /opt && \
    echo "===> install Kaldi GStreamer Server"  && \
    git clone https://github.com/alumae/kaldi-gstreamer-server.git && \
    rm -rf /opt/kaldi-gstreamer-server/.git/ && \
    rm -rf /opt/kaldi-gstreamer-server/test/



#----- Cleaning up package manager
RUN apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

##################### INSTALLATION END #####################

#--- default models

RUN mkdir -p /opt/models && cd /opt/models && \
    wget https://phon.ioc.ee/~tanela/tedlium_nnet_ms_sp_online.tgz && \
    tar -zxvf tedlium_nnet_ms_sp_online.tgz && \
    wget https://raw.githubusercontent.com/alumae/kaldi-gstreamer-server/master/sample_english_nnet2.yaml -P /opt/models && \
    find /opt/models/ -type f | xargs sed -i 's:test:/opt:g' && \
    sed -i 's:full-post-processor:#full-post-processor:g' /opt/models/sample_english_nnet2.yaml

#wget wget https://raw.githubusercontent.com/skoocda/kaldi-gstreamer-server/master/multi_hypothesis.yaml


#--- 

COPY start.sh stop.sh /opt/

RUN chmod +x /opt/start.sh && \
    chmod +x /opt/stop.sh

WORKDIR /opt/

EXPOSE 8080

CMD ["/usr/bin/supervisord"] 

#ENTRYPOINT ["/bin/bash", "/opt/start.sh", "-y", "/opt/models/sample_english_nnet2.yaml"]
#bash /opt/start.sh -y /opt/models/nnet2.yaml
#/opt/start.sh -y /opt/models/nnet2.yaml -m master-server.com -p 8888
