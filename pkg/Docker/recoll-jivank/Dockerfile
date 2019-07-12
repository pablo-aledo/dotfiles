FROM debian:jessie
MAINTAINER Jivan Kulkarni <jivank@gmail.com>

RUN echo deb http://www.lesbonscomptes.com/recoll/debian/ unstable main > \
	/etc/apt/sources.list.d/recoll.list

RUN echo deb-src http://www.lesbonscomptes.com/recoll/debian/ unstable main >> \
	/etc/apt/sources.list.d/recoll.list

RUN apt-get update && \
    apt-get install -y --force-yes recoll python-recoll python git wv poppler-utils && \
    apt-get clean

RUN mkdir /data && mkdir /root/.recoll
RUN echo topdirs = /data >> /root/.recoll/recoll.conf

RUN git clone https://github.com/koniu/recoll-webui.git
ADD start.sh /root/
ADD bgindex.sh /root/

VOLUME /data
EXPOSE 8080

RUN chmod +x /root/start.sh && chmod +x /root/bgindex.sh
CMD ["/root/start.sh"]
