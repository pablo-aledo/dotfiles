pkg install libtool
pkg install byacc
pkg install bison
pkg install flex
pkg install openssl-devel

mksrcdir /usr/src/thrift093/; cd /usr/src/thrift093/
wget https://github.com/apache/thrift/archive/0.9.3.tar.gz
tar -xvzf 0.9.3.tar.gz
cd thrift-0.9.3

./bootstrap.sh
./configure
make
sudo make install
