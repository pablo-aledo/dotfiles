pkg install libtool
pkg install byacc
pkg install bison
pkg install flex
pkg install openssl-devel

mksrcdir /usr/src/thrift/; cd /usr/src/thrift/
git clone http://github.com/apache/thrift .

./bootstrap.sh
./configure
make
sudo make install
