#sudo apt install -y html2md

cd /tmp/
wget https://github.com/suntong/html2md/releases/download/v1.5.0/html2md_1.5.0_linux_amd64.tar.gz
tar -xvf html2md_1.5.0_linux_amd64.tar.gz
sudo mv -v html2md_1.5.0_linux_amd64/html2md /usr/local/bin/
rm -fv html2md_1.5.0_linux_amd64.tar.gz
rmdir -v html2md_1.5.0_linux_amd64
