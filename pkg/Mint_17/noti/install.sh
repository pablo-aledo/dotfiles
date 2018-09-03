sudo apt-get install golang
export GOPATH=$HOME/gopath
mkdir $GOPATH
go get -u github.com/variadico/noti/cmd/noti
echo 'export GOPATH=$HOME/gopath'    >> ~/.paths
echo 'export PATH=$GOPATH/bin:$PATH' >> ~/.paths
source ~/.paths
