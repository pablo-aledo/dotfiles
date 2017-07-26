sudo apt-get install -y g++ 
sudo apt-get install -y libgtk2.0-dev
cd /tmp
wget https://github.com/Itseez/opencv/archive/3.1.0.zip
unzip 3.1.0.zip 
cd opencv-3.1.0 
cmake .
make
sudo make install
