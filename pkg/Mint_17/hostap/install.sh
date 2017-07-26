sudo apt-get install -y hostapd udhcpd

sudo stop network-manager

sudo install   -m   644   udhcpd.conf  /etc/udhcpd.conf
sudo install   -m   644   udhcpd       /etc/default/udhcpd
sudo install   -m   644   interfaces   /etc/network/interfaces
cat hostapd.conf | sed s/%pword%/`pword`/g > /tmp/hostapd.conf
sudo install   -m   644   /tmp/hostapd.conf /etc/hostapd/hostapd.conf
sudo install   -m   644   hostapd      /etc/default/hostapd
sudo install   -m   644   sysctl.conf  /etc/sysctl.conf

sudo ifconfig wlan0 192.168.42.1

sudo sh -c "echo 1 > /proc/sys/net/ipv4/ip_forward"
sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
sudo iptables -A FORWARD -i eth0 -o wlan0 -m state --state RELATED,ESTABLISHED -j ACCEPT
sudo iptables -A FORWARD -i wlan0 -o eth0 -j ACCEPT

sudo service hostapd restart
sudo service udhcpd restart

