cd
wget https://www.reaper.fm/files/7.x/reaper716_linux_x86_64.tar.xz
tar -xf reaper716_linux_x86_64.tar.xz
cd reaper_linux_x86_64
./install-reaper.sh

sudo apt-get -y install xorg-dev libasound2-dev libjack-dev
git clone https://github.com/asb2m10/dexed.git
cd dexed
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DJUCE_COPY_PLUGIN_AFTER_BUILD=TRUE
cmake --build .
