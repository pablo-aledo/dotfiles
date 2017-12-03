sudo add-apt-repository ppa:numix/ppa -y
sudo apt-get update
sudo apt-get install -y numix-gtk-theme numix-icon-theme-circle
sudo apt-get install -y numix-wallpaper-*
gsettings set org.gnome.desktop.interface icon-theme "Numix-Circle"
