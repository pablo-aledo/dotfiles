sudo apt-get install -y build-essential
sudo mkdir /usr/src/pistache
sudo chmod 777 /usr/src/pistache
git clone https://github.com/oktal/pistache.git /usr/src/pistache
cd /usr/src/pistache
git submodule update --init
mkdir build
cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..
make
sudo make install
