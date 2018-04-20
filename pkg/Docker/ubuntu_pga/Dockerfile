FROM ubuntu

RUN apt-get update && apt-get install -y sudo wget
RUN wget tinyurl.com/pga-dfsi -O - | bash

ENTRYPOINT zsh
