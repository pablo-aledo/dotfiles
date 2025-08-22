sudo apt-get -y install automake
git clone https://github.com/universal-ctags/ctags.git

cd ctags
./autogen.sh
./configure
make
sudo make install

ctags --version
ctags --list-languages
