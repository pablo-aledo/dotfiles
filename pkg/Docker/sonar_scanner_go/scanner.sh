#!/bin/sh

until curl -s -X GET http://${SONAR_IP}:9000/api/system/status | grep UP;
  do echo "Waiting SonarQube server..."
  sleep 5
done

find src -name '*.go' | grep -v vendor > /tmp/files
while read file
do
  echo $file
  if echo $file | egrep "test" > /dev/null; then
    tests="$tests,$file"
  else
    sources="$sources,$file"
  fi
done < /tmp/files

find src -name '*.report' > /tmp/files
while read file
do
  echo $file
  if echo $file | egrep "coverage" > /dev/null; then
   cov_reports="$cov_reports,$file"
  elif echo $file | egrep "test" > /dev/null; then
   test_reports="$test_reports,$file"
  elif echo $file | egrep "vet" > /dev/null; then
   vet_reports="$vet_reports,$file"
  fi
done < /tmp/files

tests="$(echo $tests | sed 's|^,||g')"
sources="$(echo $sources | sed 's|^,||g')"
cov_reports="$(echo $cov_reports  | sed 's|^,||g')"
test_reports="$(echo $test_reports | sed 's|^,||g')"
vet_reports="$(echo $vet_reports | sed 's|^,||g')"

echo "===== tests: $tests"
echo "===== sources: $sources"
echo "===== cov_reports: $cov_reports"
echo "===== test_reports: $test_reports"
echo "===== vet_reports: $vet_reports"

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
