go get github.com/dexidp/dex
cd $GOPATH/src/github.com/dexidp/dex
make

tmux new -s server -d ./bin/dex serve examples/config-dev.yaml
tmux new -s client -d ./bin/example-app
