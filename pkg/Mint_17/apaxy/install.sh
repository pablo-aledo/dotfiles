sudo cp apache2.conf /etc/apache2/apache2.conf
cd /var/www/html/
wget https://github.com/AdamWhitcroft/Apaxy/archive/master.zip
unzip master.zip
cp -r apaxy-master/apaxy/* .
mv htaccess.txt .htaccess
mv theme/htaccess.txt theme/.htaccess
sudo /etc/init.d/apache2 restart
