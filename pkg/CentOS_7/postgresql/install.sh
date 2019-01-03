sudo yum install -y centos-release-scl yum-utils
sudo yum-config-manager --enable rhel-server-rhscl-7-rpms
sudo yum install -y rh-postgresql10

scl enable rh-postgresql10 bash
sudo /opt/rh/rh-postgresql10/root/usr/bin/postgresql-setup --initdb
sudo service rh-postgresql10-postgresql start
#scl enable rh-postgresql10 bash
#initdb -U root -W datadir
#pg_ctl -D datadir -l logfile start
#createuser -U root postgres -W
#psql -U root postgres 
#alter user postgres createdb;
#alter user postgres with superuser;
