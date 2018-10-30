sudo yum install -y centos-release-scl
sudo yum-config-manager --enable rhel-server-rhscl-7-rpms
sudo yum install rh-postgresql10

scl enable rh-postgresql10 bash
sudo /opt/rh/rh-postgresql10/root/usr/bin/postgresql-setup --initdb
service rh-postgresql10-postgresql start
