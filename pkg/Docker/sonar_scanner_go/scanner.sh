#!/bin/sh

until curl -s -X GET http://${SONAR_IP}:9000/api/system/status | grep UP;
  do echo "Waiting SonarQube server..."
  sleep 5
done

find . -name '*.go' | while read file
do
  if echo $file | egrep "test" > /dev/null; then
    tests="$tests,$file"
  else
    sources="$sources,$file"
  fi
done

/usr/share/sonar-scanner-3.0.3.778-linux/bin/sonar-scanner \
  -D sonar.host.url=http://${SONAR_IP}:9000 \
  -D sonar.projectKey=${PROJECT_NAME} \
  -D sonar.projectName=${PROJECT_NAME} \
  -D sonar.projectVersion=1.0 \
  -D sonar.sources=$sources \
  -D sonar.tests=$tests \
  -D sonar.exclusions=**/*.pb.go,**/vendor/**,**/testdata/* \
  -D sonar.cxx.cppcheck.reportPath=./.cppcheck.xml \
  -D sonar.cxx.valgrind.reportPath=./.valgrind.*.xml \
  -D sonar.cxx.xunit.reportPath=./.gtest.*.xml \
  -D sonar.cxx.coverage.reportPath=./.gcov.xml \
  -D sonar.cxx.jsonCompilationDatabase=/workdir/compile_commands.json \
  -D sonar.golint.reportPath=./.golint.xml \
  -D sonar.coverage.reportPath=./.gocov.xml \
  -D sonar.test.reportPath=./.gotest.xml \
  -X
