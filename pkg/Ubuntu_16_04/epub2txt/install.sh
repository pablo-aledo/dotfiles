sudo apt-get -y install make
cd /tmp
wget https://github.com/kevinboone/epub2txt2/archive/master.zip
unzip master.zip
cd epub2txt2-master
make
sudo make install
