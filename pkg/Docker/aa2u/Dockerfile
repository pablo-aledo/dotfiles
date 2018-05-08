FROM       alpine

RUN    apk add --no-cache -q \
           ghc \
           cabal \
           git \
           alpine-sdk

RUN    cd /tmp && \
       git clone https://github.com/fmthoma/ascii-art-to-unicode.git && \
       cd ascii-art-to-unicode && \
       cabal update && \
       cabal install




