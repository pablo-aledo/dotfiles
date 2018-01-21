sudo apt-get install -y apache2 libapache2-mod-php5 mysql-client mysql-server php5-mysql
cd /var/www/html
wget http://downloads.oneye-project.org/oneye_0.9.0.zip
unzip ~/oneye_0.9.0.zip
mv oneye/* .
rm -fr oneye oneye_0.9.0.zip
chmod 777 index.html ./ ./installer package.eyepackage
