# useradd pabloaledo
# passwd pabloaledo
# nano /etc/sudoers
# mkdir /home/pabloaledo
# chown pabloaledo /home/pabloaledo/
# su -l pabloaledo

myshell zsh

pkg install docker

pkg Update

pkg install vim
pkg install w3m
pkg install tmux
pkg install novnc4
pkg install terminator
pkg install pcmanfm
pkg install ranger
pkg install mutt
pkg install xdotool
pkg install scrot
pkg install xautomation
pkg install xclip
pkg install ncdu
pkg install ffmpeg
pkg install yt-dlp
pkg install sshfs
pkg install binutils
pkg install bc
pkg install unrar
pkg install detox
pkg install pdftotext
pkg install attr
pkg install fzf z
pkg install parallel
pkg install xorg-xrandr
pkg install net-tools
pkg install xorg-xhost
pkg install rsync
pkg install poppler
pkg install socat

pkg install binutils
pkg install python-pip
pip install numpy
pip install scipy

pkg install yay
pkg install google-chrome
pkg install rsstail

sed -i 's/set \$mod Mod4/set \$mod Mod1/g' ~/.i3/config
sed -i 's/setxkbmap es/setxkbmap us/g' ~/.i3/config
sed -i '/Mod1+F2/d' ~/.i3/config
sed -i '/Mod1+Tab/d' ~/.i3/config
sed -i '/Mod1+Shift+Tab/d' ~/.i3/config
sed -i 's/bindsym Mod1+Return exec/# bindsym Mod1+Return exec/g' ~/.i3/config
sed -i 's/bindsym \$mod+Shift+Return exec --no-startup-id pcmanfm/bindsym \$mod+Shift+Return exec google-chrome/g' ~/.i3/config

sudo ln -s /usr/bin/terminator /usr/bin/gnome-terminal
sudo ln -s /usr/bin/pcmanfm /usr/bin/nautilus
sudo ln -s /usr/sbin/google-chrome-stable /usr/bin/google-chrome
[ ! -e ~/.config/terminator/config ] && rm -f ~/.config/terminator/config && ln -s ~/.dotfiles/link/.config/terminator/config ~/.config/terminator/config

sudo localectl set-locale LANG=en_US.UTF-8

i3-msg reload

sudo sysctl -w vm.min_free_kbytes=135168
sudo sysctl -w vm.swappiness=5

sudo sed -i 's/#user_allow_other/user_allow_other/g' /etc/fuse.conf

echo 'wallpaper >/dev/null 2>/dev/null' >> ~/.shell
echo 'adapt_swappiness >/dev/null 2>/dev/null' >> ~/.shell

echo 'export LOCAL_BACKUP_SRC=$HOME' >> ~/.shell
echo 'export LOCAL_BACKUP_DEST=/media/removable/2TB2/homes/admin' >> ~/.shell
echo 'export LOCAL_BACKUP_TIME_MACHINE=/media/removable/2TB2/homes/admin/time_machine' >> ~/.shell
echo 'export LOCAL_BACKUP_BAK_DIR=/media/removable/2TB2/homes/admin/rsync_bak' >> ~/.shell
source ~/.shell
