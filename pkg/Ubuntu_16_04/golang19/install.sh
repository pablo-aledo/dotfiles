wget https://storage.googleapis.com/golang/go1.9.2.linux-amd64.tar.gz -O - | sudo tar -zxv -C /usr/local/
mkdir ~/go
echo 'export GOPATH=~/go' >> ~/.paths
echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.paths
echo 'export PATH=$GOPATH/bin:$PATH' >> ~/.paths
source ~/.paths
