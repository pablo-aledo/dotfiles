pkg install readline-devel

mksrcdir /usr/src/lua5.2; cd /usr/src/lua5.2
wget https://www.lua.org/ftp/lua-5.2.4.tar.gz
tar -xvzf lua-5.2.4.tar.gz
cd lua-5.2.4
make linux
sudo make install
