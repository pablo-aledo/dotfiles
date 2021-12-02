sudo pacman -S --noconfirm libffi
sudo pacman -S --noconfirm fakeroot
sudo pacman -S --noconfirm make
cd /tmp
git clone https://aur.archlinux.org/yay.git
cd yay
makepkg -si
yay --save --nocleanmenu --nodiffmenu
