cd /opt
sudo wget https://github.com/kernc/logkeys/archive/master.zip
sudo unzip master.zip
cd logkeys-master
sudo apt-get update
pkg install automake
sudo ./autogen.sh
cd build; sudo ../configure
sudo make
sudo make install
sudo logkeys --start --keymap=/opt/logkeys-master/keymaps/en_US_ubuntu_1204.map --output test.log
