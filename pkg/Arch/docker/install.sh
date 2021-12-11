sudo pacman -S --noconfirm docker
sudo systemctl start docker
sudo usermod -aG docker $(whoami)
