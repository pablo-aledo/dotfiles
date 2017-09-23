sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade
sudo apt-get -y install gawk wget git-core diffstat unzip build-essential chrpath libsdl1.2-dev xterm curl
sudo apt-get -y install texinfo
sudo apt-get -y install lzop
sudo apt-get -y install nfs-kernel-server
sudo apt-get -y install gcc-arm-linux-gnueabihf

mkdir ~/bin
curl http://commondatastorage.googleapis.com/git-repo-downloads/repo > ~/bin/repo
chmod a+x ~/bin/repo
echo 'PATH=${PATH}:~/bin' >> ~/.paths
export PATH=${PATH}:~/bin

sudo apt-get install xinetd tftpd tftp 

echo 'service tftp'                         | sudo tee    /etc/xinetd.d/tftp
echo '{'                                    | sudo tee -a /etc/xinetd.d/tftp
echo 'protocol        = udp'                | sudo tee -a /etc/xinetd.d/tftp
echo 'port            = 69'                 | sudo tee -a /etc/xinetd.d/tftp
echo 'socket_type     = dgram'              | sudo tee -a /etc/xinetd.d/tftp
echo 'wait            = yes'                | sudo tee -a /etc/xinetd.d/tftp
echo 'user            = nobody'             | sudo tee -a /etc/xinetd.d/tftp
echo 'server          = /usr/sbin/in.tftpd' | sudo tee -a /etc/xinetd.d/tftp
echo 'server_args     = /tftp'              | sudo tee -a /etc/xinetd.d/tftp
echo 'disable         = no'                 | sudo tee -a /etc/xinetd.d/tftp
echo '}'                                    | sudo tee -a /etc/xinetd.d/tftp 

sudo mkdir /tftp
sudo chmod -R 777 /tftp
sudo chown -R nobody /tftp
mkdir /tftp/imx6 #this is the directory which we will use during development

sudo /etc/init.d/xinetd restart

