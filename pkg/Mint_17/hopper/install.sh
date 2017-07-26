sudo apt-get install -y libblocksruntime0 libkqueue0 libpthread-workqueue0
cd /tmp
wget https://d1f8bh81yd16yv.cloudfront.net/hopperv3-3.11.14.deb
sudo dpkg -i hopperv3-3.11.14.deb
sudo ln -s /opt/hopper-v3/hopper-launcher.sh /bin/hopper
