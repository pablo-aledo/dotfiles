cd /tmp/
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb 
sudo apt-get -f install -y

sudo apt install -y -f xdg-desktop-portal-gnome

# rm -rf ~/.config/google-chrome/Singleton*
