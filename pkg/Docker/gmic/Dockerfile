FROM ubuntu

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:otto-kesselgulasch/gimp && \
    apt-get update && \
    apt-get install -y gimp gmic gimp-gmic

ENTRYPOINT gimp
