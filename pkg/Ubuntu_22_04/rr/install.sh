cd
git clone https://github.com/rr-debugger/rr.git
cd rr
git checkout 5.4.0 # change this to the latest release (DO NOT BUILD HEAD)
mkdir build
cd build
cmake ..
make -j8
sudo make install
