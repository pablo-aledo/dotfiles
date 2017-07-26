sudo apt-get install -y apache2 libapache2-mod-php5 mysql-client mysql-server php5-mysql

cd /tmp 
wget https://releases.flowplayer.org/6.0.5/flowplayer-6.0.5.zip

cd /var/www/html/ 
sudo mkdir flowplayer
cd flowplayer
sudo unzip /tmp/flowplayer-6.0.5.zip

