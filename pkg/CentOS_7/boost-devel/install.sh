cd /tmp
wget https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz -O boost.tar.gz
tar -zxf boost.tar.gz
cd boost_1_69_0
./bootstrap.sh
sudo ./b2 install

