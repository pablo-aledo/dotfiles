sudo yum install -y centos-release-scl yum-utils
sudo yum-config-manager --enable rhel-server-rhscl-7-rpms
sudo yum install -y rh-postgresql10

scl enable rh-postgresql10 bash
sudo /opt/rh/rh-postgresql10/root/usr/bin/postgresql-setup --initdb
sudo service rh-postgresql10-postgresql start
