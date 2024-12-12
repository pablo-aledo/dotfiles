FROM anapsix/alpine-java

RUN apk add --no-cache -q graphviz libxtst libxi wget
RUN wget -O /usr/bin/plantuml.jar 'https://github.com/plantuml/plantuml/releases/download/v1.2024.8/plantuml-1.2024.8.jar'

ENTRYPOINT ["java", "-jar", "/usr/bin/plantuml.jar"]
