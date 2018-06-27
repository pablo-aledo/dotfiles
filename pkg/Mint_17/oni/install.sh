sudo apt-get install software-properties-common

sudo apt-add-repository ppa:neovim-ppa/stable
sudo apt-get update
sudo apt-get install neovim

cd /tmp
sudo apt-get install libappindicator1 gconf2 gconf-service libnotify4 libxtst6 libnss3 libxss1 libasound2
wget https://github.com/onivim/oni/releases/download/v0.3.6/Oni-0.3.6-amd64-linux.deb
sudo dpkg -i Oni-0.3.6-amd64-linux.deb
