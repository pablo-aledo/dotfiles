sudo apt-get install -y gcc make
cd /tmp
wget https://www.noip.com/client/linux/noip-duc-linux.tar.gz
tar -xvzf noip-duc-linux.tar.gz
cd noip-2.1.9-1
make
sudo make install
sudo /usr/local/bin/noip2
