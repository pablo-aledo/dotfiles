pkg install autoconf
pkg install libtool

sudo mkdir -p /usr/src/protobuf351 && sudo chmod 777 /usr/src/protobuf351
cd /usr/src/protobuf351
wget https://github.com/google/protobuf/archive/v3.5.1.tar.gz
tar -xvzf v3.5.1.tar.gz
cd protobuf-3.5.1

./autogen.sh
./configure
make
#make check
sudo make install
sudo ldconfig
