sudo apt-get install -y mailutils
sudo sed -i 's/inet_interfaces = all/inet_interfaces = loopback-only/g' /etc/postfix/main.cf
sudo sed -i 's/inet_protocols = all/inet_protocols = ipv4/g' /etc/postfix/main.cf
sudo service postfix restart

echo "This is a test" | mail pablo.aledo@gmail.com
