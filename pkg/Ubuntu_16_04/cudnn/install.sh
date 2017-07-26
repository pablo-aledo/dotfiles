cd
wget -x --load-cookies $COOKIES_FOLDER/nvidia.cookie 'https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v5.1/prod/8.0/cudnn-8.0-linux-x64-v5.1-tgz' -O cudnn-8.0-linux-x64-v5.1.tgz
tar -xvzf cudnn-8.0-linux-x64-v5.1.tgz
cd cuda
sudo cp lib64/* /usr/local/cuda/lib64/
sudo cp include/cudnn.h /usr/local/cuda/include/
