FROM alpine

RUN apk add --no-cache -q bash sudo findutils grep
RUN wget tinyurl.com/pga-dfsi -O - | bash

ENTRYPOINT zsh
