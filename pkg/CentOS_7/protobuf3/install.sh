pkg install autoconf
pkg install libtool

sudo mkdir -p /usr/src/protobuf && sudo chmod 777 /usr/src/protobuf
cd /usr/src/protobuf
git clone https://github.com/google/protobuf.git .
./autogen.sh
./configure
make
#make check
sudo make install
sudo ldconfig
