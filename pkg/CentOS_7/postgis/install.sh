sudo yum install -y "https://download.postgresql.org/pub/repos/yum/10/redhat/rhel-7-x86_64/pgdg-centos10-10-2.noarch.rpm"
sudo yum install -y "https://timescalereleases.blob.core.windows.net/rpm/timescaledb-0.12.1-postgresql-10-0.x86_64.rpm"
sudo yum install -y postgis25_10

sudo /usr/pgsql-10/bin/postgresql-10-setup initdb

echo "shared_preload_libraries = 'timescaledb'" | sudo tee -a /var/lib/pgsql/10/data/postgresql.conf

sudo systemctl start postgresql-10.service
