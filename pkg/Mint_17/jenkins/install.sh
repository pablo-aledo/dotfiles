sudo apt-get install -y apache2 libapache2-mod-php5 mysql-client mysql-server php5-mysql
sudo apt-get install -y jenkins


sudo rm -rf /var/lib/jenkins
sudo chown -R jenkins /var/lib/jenkins/
sudo chgrp -R jenkins /var/lib/jenkins/
sudo /etc/init.d/jenkins restart

