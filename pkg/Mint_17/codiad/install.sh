sudo apt-get install -y apache2 libapache2-mod-php5 mysql-client mysql-server php5-mysql
cd /var/www/html
sudo chmod 777 .
wget https://github.com/Codiad/Codiad/archive/v.2.6.6.tar.gz -O - | sudo tar -xvz
sudo mv Codiad-v.2.6.6 codiad
cd codiad
sudo cp config.example.php config.php
sudo chmod 777 plugins themes workspace data config.php
