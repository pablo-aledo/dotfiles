cd /tmp
wget https://www.gitkraken.com/download/linux-deb -O linux-deb
links linux-deb | grep release | grep amd64 | head -n1 > dl
(echo -n 'wget '; cat dl) | bash
sudo dpkg -i gitkraken*.deb
