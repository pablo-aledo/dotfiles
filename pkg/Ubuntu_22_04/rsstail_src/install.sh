sudo apt-get install libmrss0-dev

cd /tmp
git clone https://github.com/folkertvanheusden/rsstail.git
cd rsstail
sed -i -e "s/-liconv_hook //" Makefile
make
sudo make install
