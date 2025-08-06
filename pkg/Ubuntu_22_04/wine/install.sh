sudo dpkg --add-architecture i386
dpkg --print-foreign-architectures

sudo mkdir -pm755 /etc/apt/keyrings
wget -O - https://dl.winehq.org/wine-builds/winehq.key | sudo gpg --dearmor -o /etc/apt/keyrings/winehq-archive.key -

sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/jammy/winehq-jammy.sources

sudo apt update

sudo apt install --install-recommends winehq-stable
#sudo apt install --install-recommends winehq-devel
#sudo apt install --install-recommends winehq-staging
#apt-cache policy winehq-stable
#sudo apt install winehq-stable=7.12~jammy-1
sudo apt install winetricks

#wine --version
#wine winecfg
#WINEARCH=win64 WINEPREFIX=$HOME/.wine64 WINEDEBUG=-all winecfg
#wine clock
#wine iexplore
#msiexec -i installer_name.msi
