sudo pacman -S --noconfirm pkgconf
cd /tmp
git clone https://aur.archlinux.org/rsstail.git
cd rsstail
makepkg -si --noconfirm
