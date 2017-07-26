
cd /opt/
wget https://www.sourcetrail.com/downloads/0.12.25/linux/64bit -O - | sudo tar -xvz
echo 'export PATH=/opt/Sourcetrail:$PATH' >> ~/.paths
export PATH=/opt/Sourcetrail:$PATH
