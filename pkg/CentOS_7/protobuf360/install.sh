pkg install zlib-devel
pkg install g++

sudo mkdir -p /usr/src/protobuf360 && sudo chmod 777 /usr/src/protobuf360
cd /usr/src/protobuf360
wget https://github.com/google/protobuf/archive/v3.6.0.tar.gz
tar -xvzf v3.6.0.tar.gz
cd protobuf-3.6.0

cd cmake
mkd build
cmake3 -Dprotobuf_BUILD_TESTS=OFF ..
make
