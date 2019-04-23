go get github.com/golang/mock/gomock

mockgen -destination=src/$project/$component/mocks/$interface.go -package mocks $project/$component/interfaces $InterFace

for a in \
    $GOPATH/src/$project/$component
do
    echo "===== Testing $a ====="
    cd $a
    go test -v -cover -coverprofile=coverage.report -json | tee test.report
    go tool cover -html=coverage.report -o coverage.html
done

