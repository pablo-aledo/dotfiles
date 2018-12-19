FROM ubuntu

RUN apt-get update && apt-get install -y sudo wget
RUN wget tinyurl.com/pga-dfsi -O - | bash

run DEBIAN_FRONTEND=noninteractive apt-get install tzdata

RUN sudo apt-get install -y software-properties-common     &&\
    sudo apt-add-repository -y ppa:jonathonf/vim           &&\
    sudo apt-get update -y                                 &&\
    sudo apt-get install -y exuberant-ctags vim-nox

ENTRYPOINT zsh
