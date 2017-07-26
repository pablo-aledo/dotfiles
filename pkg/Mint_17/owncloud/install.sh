sudo apt-get install -y apache2 libapache2-mod-php5 mysql-client mysql-server php5-mysql php5-gd php5-curl

cd /var/www/html
wget https://download.owncloud.org/community/owncloud-8.2.0.tar.bz2 -O - | sudo tar -xj
sudo /etc/init.d/apache2 restart
sudo mkdir owncloud/data
sudo chmod 0770 owncloud/data
sudo chown www-data owncloud
sudo chown www-data owncloud/data
