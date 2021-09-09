sudo apt-get install make
cd /tmp
wget http://www.no-ip.com/client/linux/noip-duc-linux.tar.gz
tar xzf noip-duc-linux.tar.gz
cd noip-2.1.9-1
make
sudo make install
sudo /usr/local/bin/noip2
