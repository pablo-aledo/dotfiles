sudo apt-get install golang
export GOPATH=$HOME/gopath
mkdir $GOPATH
go get github.com/ericchiang/pup
echo 'export GOPATH=$HOME/gopath'    >> ~/.paths
echo 'export PATH=$GOPATH/bin:$PATH' >> ~/.paths
source ~/.paths
