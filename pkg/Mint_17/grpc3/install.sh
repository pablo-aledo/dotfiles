sudo apt-get install build-essential autoconf libtool

sudo mkdir -p /usr/src/grpc && sudo chmod 777 /usr/src/grpc
cd /usr/src/grpc
git clone https://github.com/grpc/grpc.git .
git submodule update --init
make
sudo make install
