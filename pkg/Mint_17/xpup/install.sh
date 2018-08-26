sudo apt-get install golang
export GOPATH=$HOME/gopath
mkdir $GOPATH
go get github.com/ericchiang/xpup
echo path $GOPATH/bin >> ~/.paths
source ~/.paths
