cd
git clone https://github.com/Kron4ek/wine-tkg.git
cd wine-tkg

#sudo dpkg --add-architecture i386
sudo apt update

#sudo apt install -y gcc-multilib g++-multilib libc6-dev-i386
#sudo apt install -y flex bison
#sudo apt install -y libx11-dev:i386 libxext-dev:i386 libxrandr-dev:i386 libxrender-dev:i386 libxcursor-dev:i386 libxi-dev:i386 libxinerama-dev:i386
#sudo apt install -y libgl1-mesa-dev:i386 libglu1-mesa-dev:i386
#sudo apt install -y libfreetype6-dev:i386 pkgconf
#./configure
sudo apt install -y gcc g++ libc6-dev
sudo apt install -y flex bison
sudo apt install -y libx11-dev libxext-dev libxrandr-dev libxrender-dev libxcursor-dev libxi-dev libxinerama-dev
sudo apt install -y libgl1-mesa-dev libglu1-mesa-dev
sudo apt install -y libfreetype6-dev pkgconf
./configure
./configure --enable-win64

make

sudo make install
sudo cp -r include/* /usr/include
