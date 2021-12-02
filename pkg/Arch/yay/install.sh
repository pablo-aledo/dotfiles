sudo pacman -s libffi
cd /tmp
git clone https://aur.archlinux.org/yay.git
cd yay
makepkg -si
yay --save --nocleanmenu --nodiffmenu
