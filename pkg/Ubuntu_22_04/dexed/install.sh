cd
sudo apt-get -y install xorg-dev libasound2-dev libjack-dev
git clone https://github.com/asb2m10/dexed.git
cd dexed
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DJUCE_COPY_PLUGIN_AFTER_BUILD=TRUE
cmake --build .
