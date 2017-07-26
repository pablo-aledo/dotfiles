#!/bin/bash 

DIR=$(dirname $(readlink -f $0))

sudo apt-get install -y ssh
sudo install -m 640 $DIR/shadow /etc/shadow
#sudo sed -i 's/Port 22/Port 5060/g' /etc/ssh/sshd_config
sudo /etc/init.d/ssh restart
sudo apt-get install -y ssh
