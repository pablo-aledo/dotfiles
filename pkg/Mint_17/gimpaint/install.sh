sudo apt-get install libgimp2.0-dev
cd
git clone https://github.com/martinjrobins/inpaintGimpPlugin.git
cd inpaintGimpPlugin
./configure
make
sudo make install
./configure --enable-user-install
make
make install
