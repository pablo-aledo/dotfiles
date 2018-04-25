#Subsonic StandAlone Docker Image
FROM ubuntu:14.04
RUN echo "Subsonic StandAlone Docker Image (v6.0)"
RUN locale-gen en_US.UTF-8  
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install openjdk-7-jre-headless wget -y

RUN mkdir -p /opt/subsonic
RUN wget -qO- http://subsonic.org/download/subsonic-6.0-standalone.tar.gz | tar xvz -C /opt/subsonic

ENV LANG en_US.UTF-8  
ENV LC_ALL en_US.UTF-8

VOLUME ["/var/music", "/var/subsonic"]
CMD /bin/bash -c "export LANG=en_US.UTF-8 && /opt/subsonic/subsonic.sh && /bin/bash"

EXPOSE 4040
