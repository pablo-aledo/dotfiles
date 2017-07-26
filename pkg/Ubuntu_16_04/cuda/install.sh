# sudo apt install linux-virtual
# sudo apt purge 'linux*aws'
# sudo reboot
# uname -r
cd
sudo apt-get update
sudo apt-get install -y build-essential
sudo apt-get install -y linux-image-extra-virtual
sudo apt-get install -y linux-headers-`uname -r`

wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run

sudo sh cuda_8.0.44_linux-run
#sudo sh cuda_8.0.44_linux-run --extract=/home/ubuntu/cuda_extract

echo 'export PATH=/usr/local/cuda-8.0/bin:$PATH'                         >> ~/.paths
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH' >> ~/.paths
echo 'blacklist nouveau'         | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo 'blacklist lbm-nouveau'     | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
echo 'options nouveau modeset=0' | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
echo 'alias nouveau off'         | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
echo 'alias lbm-nouveau off'     | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
echo options nouveau modeset=0   | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
sudo update-initramfs -u

#sudo reboot

# nomodeset nouveau.modeset=0
#cd ~/NVIDIA_CUDA-8.0_Samples/1_Utilities/deviceQuery
#make
#./deviceQuery

