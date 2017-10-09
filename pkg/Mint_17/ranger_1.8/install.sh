cd /usr/share
wget http://nongnu.org/ranger/ranger-stable.tar.gz -O - | sudo tar -xvz
cd ranger-*
sudo ./setup.py install
