[ -e /usr/bin/google-chrome ] || echo 'sudo apt-get remove -y google-chrome-stable' > ../../../uninstall/google-chrome

cd /tmp/
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt-get install -y libcurl3
sudo dpkg -i google-chrome-stable_current_amd64.deb 

sudo apt-get update
sudo apt-get install -y libnss3-tools
