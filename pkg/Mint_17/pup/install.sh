sudo apt-get install golang
export GOPATH=$HOME/gopath
mkdir $GOPATH
go get github.com/ericchiang/pup
echo path $GOPATH/bin >> ~/.paths
source ~/.paths
