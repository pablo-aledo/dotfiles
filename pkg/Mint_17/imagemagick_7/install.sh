sudo apt-get install -y libjpeg-dev build-essential
cd
wget https://www.imagemagick.org/download/ImageMagick.tar.gz
tar xvzf ImageMagick.tar.gz
cd ImageMagick-7.0.5-10
./configure
make
sudo make install
