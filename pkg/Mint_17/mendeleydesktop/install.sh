cd /tmp
wget https://www.mendeley.com/repositories/ubuntu/stable/amd64/mendeleydesktop-latest -O mendeley.deb
sudo dpkg -i mendeley.deb
mkdir -p ~/.local/share/data/
[ -d /media/DATA/Mendeley ] && sudo ln -s /media/DATA/Mendeley ~/.local/share/data/Mendeley\ Ltd.
