sudo apt-get install -y apache2 libapache2-mod-php5 mysql-client mysql-server php5-mysql
cd /var/www/html
sudo wget https://github.com/thibaud-rohmer/PhotoShow/archive/master.zip
sudo unzip master.zip
sudo mv PhotoShow-master photoshow
cd photoshow
sudo sed -i 's/path_to_your_photos_dir_goes_here/Photos/g' config.php
sudo sed -i 's/path_where_photoshow_generates_files_goes_here/Thumbs/g' config.php
sudo mkdir Photos; sudo chmod 777 Photos
sudo mkdir Thumbs; sudo chmod 777 Thumbs

