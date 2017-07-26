sudo apt-get install -y apache2 libapache2-mod-php5 mysql-client mysql-server php5-mysql aria2
cd /var/www/html
sudo wget https://github.com/ziahamza/webui-aria2/archive/master.zip
sudo unzip master.zip
sudo mv webui-aria2-master aria2
aria2c --enable-rpc --rpc-listen-all
