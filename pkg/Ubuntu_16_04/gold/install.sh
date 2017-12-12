sudo update-alternatives --install "/usr/bin/ld" "ld" "/usr/bin/ld.gold" 20
sudo update-alternatives --install "/usr/bin/ld" "ld" "/usr/bin/ld.bfd" 10

sudo update-alternatives --config ld

ld --version

#export CPP=cpp-5 gcc-5 g++-5
#env CXXFLAGS='-march=native -flto -fuse-linker-plugin' cmake .. -DCMAKE_BUILD_TYPE=Release
