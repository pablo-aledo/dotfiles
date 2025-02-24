curl -sSLf https://github.com/aclap-dev/vdhcoapp/releases/latest/download/install.sh | bash
~/.local/share/vdhcoapp/vdhcoapp install
sudo apt-get install -y flatpak
flatpak permission-set webextensions net.downloadhelper.coapp snap.firefox yes

sudo snap remove firefox
sudo apt-get install firefox
