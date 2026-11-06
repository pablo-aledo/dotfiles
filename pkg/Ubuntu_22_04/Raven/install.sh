sudo apt install build-essential cmake git libx11-dev ninja-build   # Debian, Ubuntu
#sudo dnf install gcc cmake git libX11-devel ninja-build             # Fedora
#sudo pacman -S base-devel cmake git libx11 ninja                    # Arch

git clone https://github.com/oscar-soirey/Raven.git
cd Raven
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure   # optional: the tests
cmake --install build --prefix install       # or: sudo cmake --install build
