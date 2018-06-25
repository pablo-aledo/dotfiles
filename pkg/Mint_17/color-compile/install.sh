cd /tmp
git clone https://github.com/chinaran/color-compile.git
cd color-compile
make
sudo make install
alias gcc="color_compile gcc"
alias g++="color_compile g++"
alias make="color_compile make"
