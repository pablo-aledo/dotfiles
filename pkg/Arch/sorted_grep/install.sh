cd /tmp/
wget https://downloads.sourceforge.net/project/sgrep/sgrep/sgrep-1.0/sgrep-1.0.tgz
tar -xvzf sgrep-1.0.tgz
cd sgrep-1.0
make
sudo cp sgrep /bin/
