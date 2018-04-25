FROM ubuntu:16.04
MAINTAINER Werner Beroux <werner@beroux.com>

# Install required packages.
RUN set -x \
 && export DEBIAN_FRONTEND=noninteractive \
 && apt update \
 && apt install -y \
        curl \
        docker.io \
        expect \
        git \
        nginx \
        php7.0-fpm \
        php7.0-json \
        php7.0-ldap \
        php7.0-mbstring \
        php7.0-xml \
        php7.0-zip \
    # Install docker-compose.
 && LATEST_DOCKER_COMPOSE_URI=$(curl -L https://github.com/docker/compose/releases/latest | grep -o '[^\"]*/docker-compose-Linux-x86_64') \
 && curl -L "https://github.com/$LATEST_DOCKER_COMPOSE_URI" > /usr/local/bin/docker-compose \
 && chmod +x /usr/local/bin/docker-compose \
    # Install S6.
 && apt install -y build-essential \

 && git clone git://git.skarnet.org/skalibs /tmp/skalibs \
 && cd /tmp/skalibs \
 && git checkout v2.3.10.0 \
 && ./configure \
 && make install \

 && git clone git://git.skarnet.org/execline /tmp/execline \
 && cd /tmp/execline \
 && git checkout v2.1.5.0 \
 && ./configure \
 && make install \

 && git clone git://git.skarnet.org/s6 /tmp/s6 \
 && cd /tmp/s6 \
 && git checkout v2.3.0.0 \
 && ./configure \
 && make install \

 && cd / \
 && apt-get purge --auto-remove -y build-essential \
 && rm -rf /tmp/* \
    # Clean -up
 && apt clean \
 && rm -rf /var/lib/apt/lists/* \
    # Create non-root user (with a randomly chosen UID/GUI).
 && adduser john --system --uid 2743 --group --home /code/workspace \
    # forward request and error logs to docker log collector
 && ln -sf /dev/stdout /var/log/nginx/access.log \
 && ln -sf /dev/stderr /var/log/nginx/error.log

# Codiad and config files.
RUN git clone https://github.com/Codiad/Codiad /default-code
COPY root /

RUN chown -R www-data /code
VOLUME /code

# Ports and volumes.
EXPOSE 80

# Remove error on collaboration on startup.
ENTRYPOINT ["/entrypoint.sh"]
CMD ["s6-svscan", "/etc/s6"]
