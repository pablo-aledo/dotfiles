sudo apt-get install build-essential cmake uuid-dev openssl libssl-dev
cd /tmp/
wget https://archive.apache.org/dist/qpid/proton/0.7/qpid-proton-0.7.tar.gz
tar xvfz qpid-proton-0.7.tar.gz
cd qpid-proton-0.7
mkdir build
cd build
cmake -DCMAKE_INSTA_PREFIX=/usr ..
sudo make install
