sudo apt-get install -y nfs-kernel-server nfs-common portmap
sudo mkdir /mnt/nfs
echo '/mnt/nfs 192.168.1.0/24(ro,no_root_squash,insecure)' |  sudo tee -a /etc/exports
sudo service nfs-server restart
