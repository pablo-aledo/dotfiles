FROM frolvlad/alpine-scala


RUN   apk add --no-cache -q curl ncurses && \
      mkdir ~/.ammonite; curl -L -o ~/.ammonite/predef.scala https://git.io/vVfyF && \
      curl -L -o /bin/ammonite https://git.io/vVfy5 && \
      chmod +x /bin/ammonite

ENTRYPOINT ammonite
