go get github.com/golang/mock/gomock

mockgen -destination=src/$project/$component/mocks/$interface.go -package mocks $project/$component/interfaces $InterFace

for a in \
    $GOPATH/src/$project/$component
do
    echo "===== Testing $a ====="
    cd $a
    go test -v -cover -coverprofile=c.out
    go tool cover -html=c.out -o coverage.html
done

