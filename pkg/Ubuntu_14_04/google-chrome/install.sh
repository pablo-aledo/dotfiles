[ -e /usr/bin/google-chrome ] || echo 'sudo apt-get remove -y google-chrome-stable' > ../../../uninstall/google-chrome

sudo apt-get install -y libcurl3 libgtk2.0-0 libnspr4 libnss3 libpango1.0-0 libxss1 fonts-liberation libappindicator1 xdg-utils

cd /tmp/
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
