sudo apt-get install -y pptpd
sudo install -m 644 pptpd.conf /etc/pptpd.conf 
cat chap-secrets | sed s/%pword%/`pword`/g > /tmp/chap-secrets
sudo install -m 600 /tmp/chap-secrets /etc/ppp/chap-secrets 
sudo install -m 644 pptpd-options /etc/ppp/pptpd-options
sudo install -m 644 ufw /etc/default/ufw

echo 1 | sudo tee /proc/sys/net/ipv4/ip_forward
sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
sudo iptables -t nat -A POSTROUTING -s 192.168.0.0/24 -o eth0 -j MASQUERADE
sudo iptables -A FORWARD -p tcp --syn -s 192.168.0.0/24 -j TCPMSS --set-mss 1356

#sudo /etc/init.d/networking restart
sudo /etc/init.d/pptpd restart
