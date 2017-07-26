cd /usr/share
wget https://launchpad.net/gcc-arm-embedded/5.0/5-2016-q2-update/+download/gcc-arm-none-eabi-5_4-2016q2-20160622-linux.tar.bz2 -O - | sudo tar -xvj
echo 'export PATH=/usr/share/gcc-arm-none-eabi-5_4-2016q2/bin/:$PATH' >> ~/.paths

#wget https://sourcery.mentor.com/public/gnu_toolchain/arm-none-eabi/arm-2014.05-28-arm-none-eabi-i686-pc-linux-gnu.tar.bz2 | sudo tar -xvj
#echo 'export PATH=/usr/share/arm-2014.05/bin:$PATH' >> ~/.paths

