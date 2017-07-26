#!/bin/bash

wget https://goo.gl/fd3zc -O ~/crouton
chmod +x ~/crouton
sudo sh ~/crouton -r trusty -t x11

#sudo sed -i 's/.*root.*ALL.*/&\npablo   ALL=(ALL) NOPASSWD:ALL/g' /mnt/stateful_partition/crouton/chroots/trusty/etc/sudoers
echo 'pablo   ALL=(ALL) NOPASSWD:ALL' | sudo tee /mnt/stateful_partition/crouton/chroots/trusty/etc/sudoers.d/pablo
echo 'chronos ALL=(ALL) NOPASSWD:ALL' | sudo tee /etc/sudoers.d/95_cros_base

cat << EOF > /mnt/stateful_partition/crouton/chroots/trusty/home/pablo/.remote_install_chrome
sudo apt-get install unzip

cd ~
wget https://github.com/pablo-aledo/dotfiles/archive/master.zip -O .dotfiles.zip
unzip .dotfiles.zip
mv dotfiles-master .dotfiles
cd .dotfiles
./install.sh

. /home/pablo/.dotfiles/source/.shell/adapt
adapt_chromebook

. /home/pablo/.dotfiles/source/.shell/pkg
export INSTALLERS_FOLDER=/home/pablo/.dotfiles/pkg
pkg install i3
pkg install vim-gnome
pkg install terminator
pkg install x2goclient
pkg install murrine-themes
pkg install elementary-icon-theme
pkg install lxappearance
pkg install unrar
pkg install sshfs
pkg install file-roller
pkg install evince
pkg install chromium-browser
pkg install xclip
pkg install cryptsetup
pkg install xdotool
EOF

chmod +x /mnt/stateful_partition/crouton/chroots/trusty/home/pablo/.remote_install_chrome

enter-chroot /home/pablo/.remote_install_chrome

echo 'alias starti3="sudo enter-chroot xinit"' >> /home/chronos/user/.bashrc
echo 'exec i3' > /home/chronos/user/.xinitrc
echo 'exec i3' > /mnt/stateful_partition/crouton/chroots/trusty/home/pablo/.xinitrc

cat << EOF >> /home/chronos/user/.vpn_amazon
vpn_amazon(){
	sudo stop shill
	sudo start shill BLACKLISTED_DEVICES=tun0
	#(sleep 5; echo 'nameserver 8.8.8.8' | sudo tee    /var/run/shill/resolv.conf) &
	#(sleep 6; echo 'nameserver 8.8.4.4' | sudo tee -a /var/run/shill/resolv.conf) &
	(sleep 10; sudo sed -i '1s/^/# new DNS\nnameserver 8.8.8.8\nnameserver 8.8.4.4\n# old DNS\n/' /var/run/shill/resolv.conf ) &
	sudo openvpn --config config.ovpn

}
EOF
echo 'source /home/chronos/user/.vpn_amazon' >> /home/chronos/user/.bashrc

mv /usr/sbin/update_engine /usr/sbin/update_engine_bak

