sudo apt-get install -y ranger w3m w3m-img caca-utils highlight atool poppler-utils transmission-cli mediainfo rxvt-unicode-256color zenity
ranger --copy-config=all
sed -i 's/set preview_images false/set preview_images true/g' ~/.config/ranger/rc.conf 
cat rc_end >> ~/.config/ranger/rc.conf
cp ./rifle.conf ~/.config/ranger/rifle.conf
