FROM ubuntu

RUN apt-get update && apt-get install -y sudo wget
RUN wget tinyurl.com/pga-dfsi -O - | bash

RUN sudo apt-get install software-properties-common     &&\
    sudo apt-get install python-software-properties     &&\
    sudo apt-add-repository ppa:jonathonf/vim           &&\
    sudo apt-get update                                 &&\
    sudo apt-get install vim

ENTRYPOINT zsh
