VERSION=24.4
curl -LO https://github.com/protocolbuffers/protobuf/releases/download/v${VERSION}/protoc-${VERSION}-linux-x86_64.zip
unzip protoc-*.zip -d protoc
sudo mv protoc/bin/protoc /usr/local/bin/
sudo mv protoc/include/* /usr/local/include/

#sudo apt install protobuf-compiler
