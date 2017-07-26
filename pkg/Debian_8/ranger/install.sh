sudo apt-get install -y ranger
ranger --copy-config=all
cp ./rc.conf ~/.config/ranger/rc.conf
sudo install -m 777 scope.sh /bin/scope.sh
