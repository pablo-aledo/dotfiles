sudo pacman -S --noconfirm libffi
sudo pacman -S --noconfirm fakeroot
sudo pacman -S --noconfirm make
sudo pacman -S --noconfirm binutils
sudo pacman -S --noconfirm gcc
sudo pacman -S --noconfirm pacman
cd /tmp
git clone https://aur.archlinux.org/yay.git
cd yay
makepkg -si
yay --save --nocleanmenu --nodiffmenu
