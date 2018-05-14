FROM            sonarqube:alpine

RUN             apk add --no-cache -q curl && \
                curl -L https://github.com/SonarOpenCommunity/sonar-cxx/releases/download/cxx-0.9.7/sonar-cxx-plugin-0.9.7.jar -o /opt/sonarqube/extensions/plugins/sonar-cxx-plugin-0.9.7.jar

