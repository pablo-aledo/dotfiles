sudo pacman -S --noconfirm libffi
sudo pacman -S --noconfirm fakeroot
sudo pacman -S --noconfirm make
sudo pacman -S --noconfirm binutils
sudo pacman -S --noconfirm gcc
sudo pacman -S --noconfirm pacman
#sudo pacman -S --noconfirm base-devel
cd /tmp
git clone https://aur.archlinux.org/yay.git
cd yay
makepkg -si --noconfirm
yay --save --nocleanmenu --nodiffmenu --answerdiff=None
