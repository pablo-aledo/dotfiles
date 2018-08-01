cd /tmp
wget https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.gz -O boost.tar.gz
tar -zxf boost.tar.gz
cd boost_1_65_1
./bootstrap.sh
sudo ./b2 install

