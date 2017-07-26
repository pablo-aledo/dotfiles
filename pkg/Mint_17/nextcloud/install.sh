sudo apt-get install -y apache2 libapache2-mod-php5 mysql-client mysql-server php5-mysql php5-curl

cd /tmp/ 
wget https://download.nextcloud.com/server/releases/nextcloud-10.0.1.tar.bz2
cd /var/www/html/ 
sudo tar -xvjf /tmp/nextcloud-10.0.1.tar.bz2

sudo mkdir /var/www/html/nextcloud/data/

sudo chown -R www-data:www-data /var/www/html/nextcloud/config/
sudo chown -R www-data:www-data /var/www/html/nextcloud/apps/
sudo chown -R www-data:www-data /var/www/html/nextcloud/data/

