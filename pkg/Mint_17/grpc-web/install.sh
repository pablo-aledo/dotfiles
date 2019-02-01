#pkg install protobuf3
sudo apt-get install -y make g++

sudo mkdir -p /usr/src/grpc-web && sudo chmod 777 /usr/src/grpc-web
cd /usr/src/grpc-web
git clone https://github.com/grpc/grpc-web .
sudo make install-plugin
