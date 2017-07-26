cd /usr/share
sudo mkdir rambox
cd rambox
sudo wget https://getrambox.herokuapp.com/download/linux_64 
sudo unzip linux_64
sudo chmod +x Rambox
sudo ln -s $PWD/Rambox /bin/rambox
