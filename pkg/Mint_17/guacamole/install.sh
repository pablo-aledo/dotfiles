sudo apt-get install -y guacamole-tomcat
sudo apt-get install -y libguac-client-ssh0 libguac-client-rdp0

# gsettings set org.gnome.desktop.wm.keybindings  panel-main-menu []
# gsettings set org.gnome.desktop.wm.keybindings  show-desktop []
mkdir ~/.vnc 
[ ! -e ~/.vnc/passwd ] && pword vnc | vncpasswd -f > ~/.vnc/passwd
cp xstartup ~/.vnc/
sudo apt-get install -y vnc4server
vncserver

#sudo apt-get install -y x11vnc
#x11vnc -passwd `pword` &

passwd
sudo apt-get install -y ssh

cat user-mapping.xml | sed s/%pword%/`pword`/g | sudo tee /etc/guacamole/user-mapping.xml 

sudo service guacd restart
sudo service tomcat6 restart
