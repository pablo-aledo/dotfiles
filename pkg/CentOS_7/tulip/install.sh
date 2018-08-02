pkg install glew-devel

mksrcdir /usr/src/tulip; cd /usr/src/tulip
git clone https://github.com/Tulip-Dev/tulip.git .
mkd build
cmake ..
make
sudo make install
