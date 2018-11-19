sudo mkdir -p /usr/src/node; sudo chmod 777 /usr/src/node; cd /usr/src/node
wget https://nodejs.org/dist/v10.13.0/node-v10.13.0-linux-x64.tar.xz
tar -xvJf node-v10.13.0-linux-x64.tar.xz
cd node-v10.13.0-linux-x64
sudo ln -s $PWD/bin/* /usr/local/bin
