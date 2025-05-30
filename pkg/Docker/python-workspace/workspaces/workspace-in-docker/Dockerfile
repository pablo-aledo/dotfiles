

ARG docker_registry=docker.io/alnoda
ARG image_tag=18.04-0.8

## Images used:
ARG BUILD_IMAGE=node:12.18.3
ARG DEPLOY_IMAGE=${docker_registry}/base-workspace:${image_tag}


################################################################################ BUILD 

ARG THEIA_VERSION=1.15.0
#ARG THEIA_VERSION=latest
#ARG THEIA_VERSION=next
FROM ${BUILD_IMAGE} 

ARG THEIA_VERSION

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y apt-utils \
    && apt-get install -y git \
    && apt-get install -y libsecret-1-dev \
    && mkdir /opt/theia

WORKDIR /opt/theia
ADD ${THEIA_VERSION}.package.json ./package.json
ARG GITHUB_TOKEN
RUN yarn --pure-lockfile && \
    NODE_OPTIONS="--max_old_space_size=4096" yarn theia build && \
    yarn theia download:plugins && \
    yarn --production && \
    yarn autoclean --init && \
    echo *.ts >> .yarnclean && \
    echo *.ts.map >> .yarnclean && \
    echo *.spec.* >> .yarnclean && \
    yarn autoclean --force && \
    yarn cache clean


################################################################################ IMAGE

FROM ${DEPLOY_IMAGE} 

USER root

RUN mkdir -p -m 777 /opt/theia \
    && cd /opt/theia && nodeenv --node=12.18.3 env && . env/bin/activate \
    && mkdir -p -m 777 /home/project \
    && apt-get install -y libsecret-1-dev \
    && rm -rf /home/docs

COPY --from=0 /opt/theia /opt/theia
COPY settings.json /home/abc/.theia/settings.json
COPY supervisord-theia.conf /etc/supervisord/

COPY mkdocs /home/docs
COPY README.md /home/docs/docs/docs.md

ENV SHELL=/bin/bash \
    THEIA_DEFAULT_PLUGINS=local-dir:/opt/theia/plugins \ 
    USE_LOCAL_GIT=true \
    HOME=/home/abc \
    PATH="/home/abc/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    NVM_DIR=/home/abc/.nvm

RUN echo "------------------------------------------------------ utils" \
    && rm -rf /home/abc/utils || true \
    && git clone https://github.com/bluxmit/alnoda-workspaces /tmp/alnoda-workspaces \
    && mv /tmp/alnoda-workspaces/utils /home/abc/ \
    && rm -rf /tmp/alnoda-workspaces \
    && echo "------------------------------------------------------ user" \
    && chown -R abc /opt/theia \
    && mkdir -p /var/log/theia && chown -R abc /var/log/theia \
    && chown -R abc /home/docs \
    && chown -R abc /home/abc/utils \
    && chown -R abc /home/abc/installed-python-packages \
    && find /home -type d | xargs -I{} chown -R abc {} \
    && find /home -type f | xargs -I{} chown abc {} 
    
USER abc