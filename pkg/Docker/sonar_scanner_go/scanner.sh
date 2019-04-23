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

find . -name '*.report' | while read file
do
  if echo $file | egrep "coverage" > /dev/null; then
   cov_reports="$cov_reports,$file"
  elif echo $file | egrep "test" > /dev/null; then
   test_reports="$test_reports,$file"
  elif echo $file | egrep "vet" > /dev/null; then
   vet_reports="$vet_reports,$file"
  fi
done

echo "\e[34m tests: $tests\e[0m"
echo "\e[34m sources: $sources\e[0m"
echo "\e[34m cov_reports: $cov_reports\e[0m"
echo "\e[34m test_reports: $test_reports\e[0m"
echo "\e[34m vet_reports: $vet_reports\e[0m"

/usr/share/sonar-scanner-3.0.3.778-linux/bin/sonar-scanner \
  -D sonar.host.url=http://${SONAR_IP}:9000 \
  -D sonar.projectKey=${PROJECT_NAME} \
  -D sonar.projectName=${PROJECT_NAME} \
  -D sonar.projectVersion=1.0 \
  -D sonar.sources=$sources \
  -D sonar.tests=$tests \
  -D sonar.exclusions=**/*.pb.go,**/vendor/**,**/testdata/* \
  -D sonar.golint.reportPath=./.golint.xml \
  -D sonar.coverage.reportPath=./.gocov.xml \
  -D sonar.test.reportPath=./.gotest.xml \
  -D sonar.go.coverage.reportPaths=${cov_reports} \
  -D sonar.go.tests.reportPaths=${test_reports} \
  -D sonar.go.govet.reportPaths=${vet_reports} \
  -X
