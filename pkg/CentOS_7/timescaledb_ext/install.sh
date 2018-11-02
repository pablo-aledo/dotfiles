yum install epel-release
yum update
yum install -y rh-postgresql10-postgresql-devel gcc git cmake3 make

ln -s /usr/bin/cmake3 /bin/cmake

git clone https://github.com/timescale/timescaledb.git
cd timescaledb
git checkout 0.8.0
./bootstrap
cd build && make
make install

