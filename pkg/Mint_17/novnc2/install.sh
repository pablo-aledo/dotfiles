mkdir ~/.vnc 
#cp passwd ~/.vnc/
#cp xstartup ~/.vnc/
#chmod +x ~/.vnc/xstartup
sudo apt-get install -y vnc4server

gsettings set org.gnome.desktop.wm.keybindings  panel-main-menu '[]'
gsettings set org.gnome.desktop.wm.keybindings  show-desktop '[]'

cd
git clone https://github.com/novnc/noVNC.git

