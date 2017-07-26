sudo apt-get install -y apache2
sudo a2enmod ssl
sudo service apache2 restart
sudo mkdir /etc/apache2/ssl
( echo US; echo New York; echo New York City; echo Your Company; echo Department of Kittens; echo your_domain.com; echo your_email@domain.com ) | sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /etc/apache2/ssl/apache.key -out /etc/apache2/ssl/apache.crt
sudo cp default-ssl.conf /etc/apache2/sites-available/default-ssl.conf
sudo a2ensite default-ssl.conf
sudo service apache2 restart
