FROM debian:jessie
MAINTAINER dsheyp

RUN echo deb http://www.lesbonscomptes.com/recoll/debian/ unstable main > \
    /etc/apt/sources.list.d/recoll.list

RUN echo deb-src http://www.lesbonscomptes.com/recoll/debian/ unstable main >> \
    /etc/apt/sources.list.d/recoll.list

RUN apt-get update && \
    apt-get install -y --force-yes recoll python-recoll python git wv poppler-utils && \
    apt-get clean

RUN apt-get install -y --force-yes unzip xsltproc unrtf untex libimage-exiftool-perl antiword pstotext

RUN mkdir /homes && mkdir /root/.recoll

RUN git clone https://github.com/dsheyp/recoll-webui.git

VOLUME /homes
EXPOSE 8080

CMD ["/usr/bin/python", "/recoll-webui/webui-standalone.py", "-a", "0.0.0.0"]
