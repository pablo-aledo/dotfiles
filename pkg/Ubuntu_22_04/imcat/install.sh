mkdir /tmp/imcat
cd /tmp/imcat
wget https://github.com/stolk/imcat/archive/refs/heads/master.zip
unzip master.zip
cd imcat-master
make
sudo make install
