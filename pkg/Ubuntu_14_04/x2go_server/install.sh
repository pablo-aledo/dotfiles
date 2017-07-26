sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:x2go/stable
sudo apt-get update
sudo apt-get install -y x2goserver x2goserver-xsession

echo '#!/bin/sh'               | sudo tee    /bin/starti3
echo 'export LANG=en_US.UTF-8' | sudo tee -a /bin/starti3
echo 'exec i3'                 | sudo tee -a /bin/starti3
sudo chmod +x /bin/starti3
