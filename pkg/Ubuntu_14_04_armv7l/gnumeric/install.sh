cd /tmp/
sudo apt-get install -y libgoffice-0.8-8
wget http://ports.ubuntu.com/ubuntu-ports/pool/universe/g/gnumeric/gnumeric_1.10.17-1ubuntu2_armhf.deb
wget http://ports.ubuntu.com/ubuntu-ports/pool/universe/g/gnumeric/gnumeric-common_1.10.17-1ubuntu2_all.deb
sudo dpkg -i gnumeric*.deb
