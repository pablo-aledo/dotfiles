FROM anapsix/alpine-java

RUN apk add --no-cache -q graphviz libxtst libxi
RUN wget -O /usr/bin/plantuml.jar 'https://sourceforge.net/projects/plantuml/files/latest/download?source=typ_redirect'

ENTRYPOINT ["java", "-jar", "/usr/bin/plantuml.jar"]
