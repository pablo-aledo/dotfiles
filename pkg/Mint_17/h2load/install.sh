cd
sudo apt install g++ make binutils autoconf automake autotools-dev libtool pkg-config zlib1g-dev libcunit1-dev libssl-dev libxml2-dev libev-dev libevent-dev -y --force-yes
git clone https://github.com/nghttp2/nghttp2.git && cd nghttp2
autoreconf -i
automake
autoconf
./configure --enable-app
make
sudo make install
