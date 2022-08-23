sudo yum install -y xorg-x11-xkb-utils i3 xorg-x11-xauth

mkdir ~/.vnc
[ ! -e ~/.vnc/passwd ] && pword vnc | vncpasswd -f > ~/.vnc/passwd
chmod 0600 ~/.vnc/passwd
cp xstartup ~/.vnc/
chmod +x ~/.vnc/xstartup

sudo mkdir /usr/share/tigervnc; sudo chmod 777 /usr/share/tigervnc/; cd /usr/share/tigervnc
wget 'https://bintray.com/tigervnc/stable/download_file?file_path=tigervnc-1.8.0.x86_64.tar.gz' -O tigervnc-1.8.0.x86_64.tar.gz
tar -xzf tigervnc-1.8.0.x86_64.tar.gz
sudo cp -r tigervnc-1.8.0.x86_64/usr/* /usr/

sudo mkdir /usr/share/noVNC; sudo chmod 777 /usr/share/noVNC/; cd /usr/share/noVNC
git clone https://github.com/novnc/noVNC.git /usr/share/noVNC
touch /usr/share/noVNC/index.html

cd; sudo rm -fr /usr/share/tigervnc

