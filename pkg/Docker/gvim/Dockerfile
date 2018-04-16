FROM debian
MAINTAINER verybada

RUN sed -i "s/http:\/\/httpredir.debian.org\/debian/http:\/\/ftp.tw.debian.org\/debian\//g" /etc/apt/sources.list
RUN apt-get update; apt-get install sudo vim vim-gnome -y
RUN export uid=1000 gid=1000 && \
	mkdir -p /home/developer && \
	echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
	echo "developer:x:${uid}:" >> /etc/group && \
	echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
	chmod 0440 /etc/sudoers.d/developer && \
	chown ${uid}:${gid} -R /home/developer
USER developer
ENV HOME=/home/developer
CMD gvim -f
