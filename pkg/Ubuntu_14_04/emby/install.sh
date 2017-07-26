wget -qO - http://download.opensuse.org/repositories/home:emby/xUbuntu_14.04/Release.key | sudo apt-key add -
sudo sh -c "echo 'deb http://download.opensuse.org/repositories/home:/emby/xUbuntu_14.04/ /' >> /etc/apt/sources.list.d/emby-server.list"
sudo apt-get update
sudo apt-get install -y emby-server
sudo service emby-server restart
