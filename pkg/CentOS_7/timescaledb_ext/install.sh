sudo yum install -y epel-release
sudo yum update
sudo yum install -y rh-postgresql10-postgresql-devel gcc git cmake3 make

sudo ln -s /usr/bin/cmake3 /bin/cmake
sudo ln -s /opt/rh/rh-postgresql10/root/usr/bin/pg_config /bin/pg_config

cd
git clone https://github.com/timescale/timescaledb.git
cd timescaledb
git checkout 0.8.0
./bootstrap
cd build && make
sudo make install

