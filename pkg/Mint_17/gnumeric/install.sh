sudo apt-get install -y libgoffice-0.8-8
cd /tmp
wget http://ftp.us.debian.org/debian/pool/main/g/gnumeric/gnumeric_1.10.17-1.1_amd64.deb
wget http://ftp.us.debian.org/debian/pool/main/g/gnumeric/gnumeric-common_1.10.17-1.1_all.deb
sudo dpkg -i gnumeric_1.10.17-1.1_amd64.deb gnumeric-common_1.10.17-1.1_all.deb
