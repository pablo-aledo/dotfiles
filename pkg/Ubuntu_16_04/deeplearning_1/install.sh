sudo apt-get update -y

sudo apt install -y linux-virtual
sudo apt purge -y 'linux*aws'
echo 'blacklist nouveau'         | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo 'blacklist lbm-nouveau'     | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
echo 'options nouveau modeset=0' | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
echo 'alias nouveau off'         | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
echo 'alias lbm-nouveau off'     | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
echo options nouveau modeset=0   | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
sudo update-initramfs -u

sudo reboot
