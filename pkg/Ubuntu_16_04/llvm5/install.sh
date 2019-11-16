sudo apt-get install -y g++ make
cd
wget http://releases.llvm.org/5.0.0/llvm-5.0.0.src.tar.xz
wget http://releases.llvm.org/5.0.0/cfe-5.0.0.src.tar.xz
tar xfv llvm-5.0.0.src.tar.xz
tar xfv cfe-5.0.0.src.tar.xz
mv cfe-5.0.0.src llvm-5.0.0.src/tools/clang
cd llvm-5.0.0.src
mkdir build
cd build
cmake ..
make
