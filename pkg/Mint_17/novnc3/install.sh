mkdir ~/.vnc 
cp passwd ~/.vnc/
cp xstartup ~/.vnc/
chmod +x ~/.vnc/xstartup

cd
wget https://bintray.com/tigervnc/stable/download_file?file_path=tigervnc-1.8.0.x86_64.tar.gz -O tigervnc-1.8.0.x86_64.tar.gz
tar -xzf tigervnc-1.8.0.x86_64.tar.gz
sudo cp -r tigervnc-1.8.0.x86_64/usr/* /usr/

cd
git clone https://github.com/novnc/noVNC.git

