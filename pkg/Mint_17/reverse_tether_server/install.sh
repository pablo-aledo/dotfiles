sudo apt-get install -y udhcpd

sudo install   -m   644   udhcpd.conf  /etc/udhcpd.conf
sudo install   -m   644   udhcpd       /etc/default/udhcpd
sudo install   -m   644   interfaces   /etc/network/interfaces
sudo install   -m   644   sysctl.conf  /etc/sysctl.conf

sudo ifconfig usb0 192.168.42.1

sudo sh -c "echo 1 > /proc/sys/net/ipv4/ip_forward"
sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
sudo iptables -A FORWARD -i eth0 -o usb0 -m state --state RELATED,ESTABLISHED -j ACCEPT
sudo iptables -A FORWARD -i usb0 -o eth0 -j ACCEPT

sudo service udhcpd restart

## In the phone
# netcfg rndis0 dhcp
# ndc resolver flushif rndis0
# ndc resolver flushdefaultif
# ndc resolver setifdns rndis0 8.8.8.8 8.8.4.4
# ndc resolver setdefaultif rndis0
