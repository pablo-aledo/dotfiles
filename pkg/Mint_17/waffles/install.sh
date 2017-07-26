sudo apt-get install -y g++ make libpng12-dev libjpeg8-dev libsdl1.2-dev freeglut3-dev libx11-dev
cd /tmp
git clone --depth=1 https://github.com/mikegashler/waffles
cd /tmp/waffles/src
sudo make install
cd /tmp/waffles/src/depends
sudo make install
cd /tmp/waffles/demos
sudo make opt
