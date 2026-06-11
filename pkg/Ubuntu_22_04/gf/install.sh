pkg install libx11-dev
cd /tmp
wget https://github.com/nakst/gf/archive/refs/heads/master.zip
unzip master.zip
cd gf-master
./build.sh
sudo cp gf2 /usr/bin
