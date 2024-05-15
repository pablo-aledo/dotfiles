cd
sudo apt install libopencv-dev
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcpu.zip
git clone https://github.com/pierotofy/OpenSplat OpenSplat
unzip $OLDPWD/banana.zip
cd OpenSplat
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch/ .. && make -j$(nproc)
./opensplat ~/banana -n 200
