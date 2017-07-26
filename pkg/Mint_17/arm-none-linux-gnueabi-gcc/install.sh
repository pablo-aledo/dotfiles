cd /tmp
wget https://sourcery.mentor.com/public/gnu_toolchain/arm-none-linux-gnueabi/arm-2009q1-203-arm-none-linux-gnueabi.bin
sudo rm /bin/sh
sudo ln -s /bin/bash /bin/sh
chmod +x arm-2009q1-203-arm-none-linux-gnueabi.bin
./arm-2009q1-203-arm-none-linux-gnueabi.bin

#sudo mkdir -p /opt/codesourcery
#sudo chmod ugo+wrx /opt/codesourcery
#sudo rm /bin/sh
#sudo ln -s /bin/bash /bin/sh
#sudo apt-get install libgtk2.0-0:i386 libxtst6:i386 gtk2-engines-murrine:i386 libstdc++6 libxt6:i386
#sudo apt-get install libdbus-glib-1-2:i386 libasound2:i386
#sudo apt-get install openjdk-6-jre
#chmod ugo+x arm-2009q1-203-arm-none-linux-gnueabi.bin
#./arm-2009q1-203-arm-none-linux-gnueabi.bin
