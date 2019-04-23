FROM            sonarqube:alpine

RUN             apk add --no-cache -q curl && \
                curl -L https://github.com/SonarOpenCommunity/sonar-cxx/releases/download/cxx-0.9.7/sonar-cxx-plugin-0.9.7.jar -o /opt/sonarqube/extensions/plugins/sonar-cxx-plugin-0.9.7.jar

RUN             curl -L https://binaries.sonarsource.com/Distribution/sonar-go-plugin/sonar-go-plugin-1.1.0.1612.jar -o /opt/sonarqube/extensions/plugins/sonar-go-plugin-1.1.0.1612.jar
