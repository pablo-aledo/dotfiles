# sudo apt install build-essential cmake pkg-config libsdl2-dev libgl-dev libasound2-dev libjack-jackd2-dev libx11-dev
#
# # Fedora: sudo dnf install cmake gcc-c++ SDL2-devel alsa-lib-devel libX11-devel mesa-libGL-devel
# # Arch:   sudo pacman -S cmake sdl2 alsa-lib libx11 mesa
# # FreeBSD: pkg install cmake ninja sdl2 libX11 sndio    OpenBSD: pkg_add cmake ninja sdl2
#
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# cmake --build build -j4           # a fixed job count: a bare -j starts every compile at once and can run out of memory
# ctest --test-dir build          # 271 tests
# ./build/gridline-app            # start the app

sudo wget https://github.com/lolitemaultes/Gridline/releases/download/v0.7.1/Gridline-0.7.1-linux-x86_64.AppImage -O /usr/bin/gridline
sudo chmod +x /usr/bin/gridline
