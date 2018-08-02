pkg install nasm

mksrcdir /usr/src/libjpeg-turbo; cd /usr/src/libjpeg-turbo
wget https://github.com/libjpeg-turbo/libjpeg-turbo/archive/master.zip
unzip master.zip
cd libjpeg-turbo-master
mkd build
cmake ..
make
sudo make install
