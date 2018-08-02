pkg install libtool
pkg install byacc
pkg install bison
pkg install flex
pkg install openssl-devel
pkg install libevent-devel
pkg install python-devel
pkg install lua5.2-devel

sudo ln -s /usr/lib64/liblua.so /usr/lib64/liblua5.2.so

mksrcdir /usr/src/thrift092/; cd /usr/src/thrift092/
wget https://github.com/apache/thrift/archive/0.9.2.tar.gz
tar -xvzf 0.9.2.tar.gz
cd thrift-0.9.2

./bootstrap.sh
./configure
make
sudo make install





