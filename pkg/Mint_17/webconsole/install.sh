
sudo apt-get install -y apache2 libapache2-mod-php5 mysql-client mysql-server php5-mysql 
cd /var/www/html 
sudo wget https://github.com/nickola/web-console/releases/download/v0.9.5/webconsole-0.9.5.zip
sudo unzip webconsole-0.9.5.zip
sudo sed -i 's/$USER =.*/$USER = "mint";/g' webconsole/webconsole.php
sudo sed -i 's/$PASSWORD =.*/$PASSWORD = "'`pword`'";/g' webconsole/webconsole.php
