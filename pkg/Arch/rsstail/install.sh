sudo pacman -S --noconfirm pkgconf fakeroot binutils gcc make libffi

cd /tmp
git clone https://aur.archlinux.org/libnxml.git
cd libnxml
makepkg -si --noconfirm

cd /tmp
git clone https://aur.archlinux.org/libmrss.git
cd libmrss
makepkg -si --noconfirm

cd /tmp
git clone https://aur.archlinux.org/rsstail.git
cd rsstail
makepkg -si --noconfirm
