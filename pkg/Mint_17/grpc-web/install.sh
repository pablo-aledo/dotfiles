sudo mkdir -p /usr/src/grpc-web && sudo chmod 777 /usr/src/grpc-web
cd /usr/src/grpc-web
git clone https://github.com/grpc/grpc-web
cd grpc-web
sudo make install-plugin
