sudo yum install -y epel-release
sudo yum install -y https://download1.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm
sudo yum update -y
sudo yum install -y rh-postgresql10-postgresql-devel gcc make wget libxml2-devel geos-devel gdal-devel

cd
mkdir proj4
cd proj4
wget http://download.osgeo.org/proj/proj-4.9.1.tar.gz
tar -xvzf proj-4.9.1.tar.gz
cd proj-4.9.1
./configure
make
make check
sudo make install


cd
wget https://download.osgeo.org/postgis/source/postgis-2.4.5.tar.gz
tar -xvzf postgis-2.4.5.tar.gz
cd postgis-2.4.5
./configure
make
sudo make install

# sudo ln -s /usr/local/lib/libproj.so.9 /usr/lib64/libproj.so.9
