mkdir ~/.vnc 
cp passwd ~/.vnc/
cp xstartup ~/.vnc/
sudo apt-get install -y vnc4server
vncserver

cd
git clone https://github.com/novnc/noVNC.git

