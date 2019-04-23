FROM            alpine

RUN             apk add --no-cache openssl openjdk8 curl unzip && \
                curl -L https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-3.0.3.778-linux.zip -o sonar-scanner-cli-3.0.3.778-linux.zip && \
                unzip sonar-scanner-cli-3.0.3.778-linux.zip -d /usr/share && \
                sed -i -e 's/use_embedded_jre=true/use_embedded_jre=false/' /usr/share/sonar-scanner-3.0.3.778-linux/bin/sonar-scanner

ADD             scanner.sh /bin/scanner.sh
RUN             chmod +x /bin/scanner.sh

ENTRYPOINT      scanner.sh
