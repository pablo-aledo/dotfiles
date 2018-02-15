sudo apt-get install build-essential autoconf libtool

mksrcdir /usr/src/grpc
cd /usr/src/grpc
git clone https://github.com/grpc/grpc.git .
git submodule update --init
make
sudo make install
