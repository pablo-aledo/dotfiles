# sudo apt-get install golang
# export GOPATH=$HOME/gopath
# mkdir $GOPATH
# go get -u github.com/variadico/noti/cmd/noti
# echo 'export GOPATH=$HOME/gopath'    >> ~/.paths
# echo 'export PATH=$GOPATH/bin:$PATH' >> ~/.paths
# source ~/.paths
cd /bin
curl -L $(curl -s https://api.github.com/repos/variadico/noti/releases/latest | awk '/browser_download_url/ { print $2 }' | grep 'linux-amd64' | sed 's/"//g') | sudo tar -xz
