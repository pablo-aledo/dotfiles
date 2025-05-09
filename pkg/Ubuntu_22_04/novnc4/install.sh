sudo apt-get install -y i3 feh i3status pcmanfm xauth x11-xkb-utils

(
sudo mkdir /usr/share/tigervnc; sudo chmod 777 /usr/share/tigervnc/; cd /usr/share/tigervnc
wget 'https://downloads.sourceforge.net/project/tigervnc/stable/1.8.0/tigervnc-1.8.0.x86_64.tar.gz' -O tigervnc-1.8.0.x86_64.tar.gz
tar -xzf tigervnc-1.8.0.x86_64.tar.gz
sudo cp -r tigervnc-1.8.0.x86_64/usr/* /usr/
cd; sudo rm -fr /usr/share/tigervnc
)

(
sudo mkdir /usr/share/noVNC; sudo chmod 777 /usr/share/noVNC/; cd /usr/share/noVNC
git clone https://github.com/novnc/noVNC.git /usr/share/noVNC
touch /usr/share/noVNC/index.html
sudo git checkout dd20b17d49a2394b586175f870e00b4b64c2817d
rm -fr /usr/share/noVNC/{.git,.github,.gitignore,.gitmodules,docs,LICENSE.txt,README.md,tests,.travis.yml,VERSION}
)

mkdir /usr/share/noVNC/utils/websockify
git clone https://github.com/novnc/websockify.git /usr/share/noVNC/utils/websockify
mkdir ~/.vnc
#[ ! -e ~/.vnc/passwd ] && pword vnc | vncpasswd -f > ~/.vnc/passwd
#chmod 0600 ~/.vnc/passwd
\cp xstartup ~/.vnc/
chmod +x ~/.vnc/xstartup
