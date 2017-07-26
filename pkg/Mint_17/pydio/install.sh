sudo apt-get install -y apache2 libapache2-mod-php5 mysql-client mysql-server php5-mysql

cd /tmp/
wget https://download.pydio.com/pub/core/archives/pydio-core-7.0.0.zip
cd /var/www/html/ 
sudo unzip /tmp/pydio-core-7.0.0.zip
sudo chown -R www-data:www-data /var/www/html/pydio-core-7.0.0/data

echo 'create database pydio;' | mysql -u root -p `pword`

echo '<Directory /var/www/html/pydio-core-7.0.0/>' | sudo tee -a /etc/apache2/apache2.conf
echo '	AllowOverride All'                         | sudo tee -a /etc/apache2/apache2.conf
echo '</Directory>'                                | sudo tee -a /etc/apache2/apache2.conf

