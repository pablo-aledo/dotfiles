mksrcdir /usr/src/protobuf
cd /usr/src/protobuf
git clone https://github.com/google/protobuf.git .
./autogen.sh
./configure
make
#make check
sudo make install
sudo ldconfig
