sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt-get update
sudo apt-get install -y nvidia-375 nvidia-settings nvidia-modprobe

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
# Verify that the key fingerprint is 9DC8 5822 9FC7 DD38 854A E2D8 8D81 803C 0EBF CD88
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
 "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
 $(lsb_release -cs) \
 stable"
sudo apt-get update
sudo apt-get install -y docker-ce

wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker_1.0.1-1_amd64.deb && rm /tmp/nvidia-docker_1.0.1-1_amd64.deb

sudo nvidia-docker run --rm nvidia/cuda nvidia-smi
