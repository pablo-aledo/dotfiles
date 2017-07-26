cd /media/DATA/delme/ 
dd if=/dev/zero of=llvm.img bs=1G count=5
sudo mkfs.ext3 llvm.img
sudo mkdir /llvm-3.4
sudo mount llvm.img /llvm-3.4
cd /llvm-3.4
sudo chmod 777 .
rm -rf *

wget http://llvm.org/releases/3.4/llvm-3.4.src.tar.gz -O - | tar -xz
mv llvm-3.4/* .
rm -rf llvm-3.4

cd tools
wget http://llvm.org/releases/3.4/clang-3.4.src.tar.gz -O - | tar -xz
mv clang-3.4 clang

cd /llvm-3.4
mkdir build
cd build 
cmake ..
make
sudo make install
