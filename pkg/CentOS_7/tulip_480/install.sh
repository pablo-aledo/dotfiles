pkg install libjpeg-turbo


mksrcdir /usr/src/tulip_480; cd /usr/src/tulip_480
wget https://github.com/Tulip-Dev/tulip/archive/tulip_4_8_0.tar.gz
tar -xvzf tulip_4_8_0.tar.gz
cd tulip-tulip_4_8_0
mkd build
cmake ..

