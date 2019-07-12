FROM debian:jessie
MAINTAINER Victor <victor@me.com>

RUN adduser --disabled-password docker

VOLUME /home/docker/data
EXPOSE 8080

#this should be picked up by recollindex
ENV RECOLL_CONFDIR /home/docker/data/.recoll 

RUN echo deb http://www.lesbonscomptes.com/recoll/debian/ unstable main > \
	/etc/apt/sources.list.d/recoll.list

RUN echo deb-src http://www.lesbonscomptes.com/recoll/debian/ unstable main >> \
	/etc/apt/sources.list.d/recoll.list

RUN apt-get update && \
	apt-get install -y --force-yes locales
# Set the locale
#http://jaredmarkell.com/docker-and-locales/
#https://stackoverflow.com/questions/28405902/how-to-set-the-locale-inside-a-ubuntu-docker-container
RUN sed -i -e 's/# de_DE.UTF-8 UTF-8/de_DE.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen de_DE.UTF-8  
ENV LANG de_DE.UTF-8  
ENV LANGUAGE de_DE:de
ENV LC_ALL de_DE.UTF-8 


RUN apt-get update && \
    apt-get install -y --force-yes recoll python-recoll python git wv poppler-utils && \
    #install german language pack for aspell
    apt-get install -y --force-yes aspell-de && \
    apt-get clean



RUN git clone https://github.com/viktor-c/recoll-webui.git -b viktor /home/docker/recoll-webui/

# Move recoll files
#bgindex.sh and startrecoll.sh
COPY scripts/ /usr/local/bin/
RUN chmod +x /usr/local/bin/startrecoll.sh && chmod +x /usr/local/bin//bgindex.sh

USER docker
COPY recoll.conf /home/docker/data/.recoll/
CMD ["/usr/local/bin/startrecoll.sh"]
