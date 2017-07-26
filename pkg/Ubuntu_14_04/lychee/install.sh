sudo apt-get install -y apache2 libapache2-mod-php5 mysql-client mysql-server php5-mysql php5-gd
cd /var/www/html
sudo wget https://github.com/electerious/Lychee/archive/master.zip
sudo unzip master.zip
sudo mv Lychee-master lychee
sudo chmod -R 777 lychee/data lychee/uploads
