sudo mkdir -p /usr/src/grpc && sudo chmod 777 /usr/src/grpc
cd /usr/src/grpc
git clone https://github.com/grpc/grpc.git .
git checkout 7e6b5db90de63c820e80440502aa1427bcd72ed9
git submodule update --init

mkd build
cmake3 ..

sed -i 's/gRPC_ZLIB_PROVIDER:STRING=.*/gRPC_ZLIB_PROVIDER:STRING=package/g' CMakeCache.txt
sed -i 's/gRPC_CARES_PROVIDER:STRING=.*/gRPC_CARES_PROVIDER:STRING=package/g' CMakeCache.txt
sed -i 's/gRPC_SSL_PROVIDER:STRING=.*/gRPC_SSL_PROVIDER:STRING=package/g' CMakeCache.txt
sed -i 's/gRPC_PROTOBUF_PROVIDER:STRING=.*/gRPC_PROTOBUF_PROVIDER:STRING=package/g' CMakeCache.txt

cmake3 ..
make
sudo make install
