echo 'deb https://download.jitsi.org stable/' | sudo tee /etc/apt/sources.list.d/jitsi-stable.list
wget -qO -  https://download.jitsi.org/jitsi-key.gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get -y install jitsi-meet
