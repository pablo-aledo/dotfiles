#!/bin/sh

config(){
	rm -rf $dbPath/*
	mkdir -p $dbPath

	cp /etc/extdb.template $conf_file
	chmod u+w $conf_file

	for a in $(ls /workdir/schemas/*.schema)
	do
		sed -i "s|#schemas#|include $a\n#schemas#|g" $conf_file
	done
}

start ()
{
    # kill existing slapd on that ip and port
    pgrep -f "slapd.*$uri" | xargs -r kill -HUP


    cd /workdir

    sleep 1
    /usr/sbin/slapd -f $conf_file -h "$uri_plusipv6" -d $trace_level > $log_file 2>&1 &
    sleep 3
}

populate_default_data ()
{

    manager_dn="cn=manager,dc=operator,dc=com"

    for a in /workdir/populate
    do
    	ldapmodify -a \
    	-r \
    	-x \
    	-c \
    	-P 3 \
    	-H $uri \
    	-D $manager_dn \
    	-w $passwd \
    	-f $a
    done

}

ip=0.0.0.0
port=9123
trace_level=1
dbPath=/workdir/database
log_file=/workdir/openLdap.log
conf_file=/workdir/extdb.conf
uri="ldap://$ip:$port"
uri_plusipv6="$uri ldap://[::0]:$port"
passwd="manager"

config
start
populate_default_data

