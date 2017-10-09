cd /usr/share
wget https://downloads.sourceforge.net/project/ispcmirror/v1.9.1/ispc-v1.9.1-linux.tar.gz -O - | sudo tar -xvz
echo 'export PATH=/usr/share/ispc-v1.9.1-linux/:$PATH' >> ~/.paths
export PATH=/usr/share/ispc-v1.9.1-linux/:$PATH
