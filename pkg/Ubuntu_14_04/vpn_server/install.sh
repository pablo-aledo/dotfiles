sudo apt-get install -y openvpn easy-rsa mutt

sudo install -m 644 server.conf /etc/openvpn/server.conf

echo 1 | sudo tee /proc/sys/net/ipv4/ip_forward
sudo sed -i 's/#net.ipv4.ip_forward=1/net.ipv4.ip_forward=1/g' /etc/sysctl.conf

sudo cp -r /usr/share/easy-rsa/ /etc/openvpn
sudo mkdir /etc/openvpn/easy-rsa/keys

sudo sed -i 's/export KEY_NAME="EasyRSA"/export KEY_NAME="server"/g'                           /etc/openvpn/easy-rsa/vars

sudo openssl dhparam -out /etc/openvpn/dh2048.pem 2048
cd /etc/openvpn/easy-rsa

echo 'cd /etc/openvpn/easy-rsa'          >  /tmp/cmd
echo '. ./vars'                          >> /tmp/cmd
echo './clean-all'                       >> /tmp/cmd
echo './build-ca --batch'                >> /tmp/cmd
echo './build-key-server --batch server' >> /tmp/cmd
sudo bash /tmp/cmd

sudo cp /etc/openvpn/easy-rsa/keys/{server.crt,server.key,ca.crt} /etc/openvpn
sudo service openvpn start
sudo service openvpn status

echo 'cd /etc/openvpn/easy-rsa'    >  /tmp/cmd
echo '. ./vars'                    >> /tmp/cmd
echo './build-key --batch client1' >> /tmp/cmd
sudo bash /tmp/cmd

sudo cp /usr/share/doc/openvpn/examples/sample-config-files/client.conf /etc/openvpn/easy-rsa/keys/client.ovpn

mkdir ~/vpn
sudo install -m 777 /etc/openvpn/easy-rsa/keys/client1.crt ~/vpn
sudo install -m 777 /etc/openvpn/easy-rsa/keys/client1.key ~/vpn
sudo install -m 777 /etc/openvpn/easy-rsa/keys/client.ovpn ~/vpn
sudo install -m 777 /etc/openvpn/ca.crt ~/vpn

[ -e /usr/local/bin/noip2 ] && IP=pabloaledo.ddns.net
[ -e /usr/local/bin/noip2 ] || IP=`wget http://ipinfo.io/ip -qO -`

cd ~/vpn
sed -i "s/remote my-server-1 1194/remote $IP 1194/g" client.ovpn
sed -i 's/;user nobody/user nobody/g'                client.ovpn
sed -i 's/;group nogroup/group nogroup/g'            client.ovpn
sed -i 's/ca ca.crt/#ca ca.crt/g'                    client.ovpn
sed -i 's/cert client.crt/#cert client.crt/g'        client.ovpn
sed -i 's/key client.key/#key client.key/g'          client.ovpn

cat client.ovpn >  config.ovpn
echo '<ca>'     >> config.ovpn
cat ca.crt      >> config.ovpn
echo '</ca>'    >> config.ovpn
echo '<cert>'   >> config.ovpn
cat client1.crt >> config.ovpn
echo '</cert>'  >> config.ovpn
echo '<key>'    >> config.ovpn
cat client1.key >> config.ovpn
echo '</key>'   >> config.ovpn

sudo ufw allow ssh
sudo sed -i 's/DEFAULT_FORWARD_POLICY="DROP"/DEFAULT_FORWARD_POLICY="ACCEPT"/g' /etc/default/ufw
echo '# START OPENVPN RULES'                              >  /tmp/insert
echo '# NAT table rules'                                  >> /tmp/insert
echo '*nat'                                               >> /tmp/insert
echo ':POSTROUTING ACCEPT [0:0] '                         >> /tmp/insert
echo '# Allow traffic from OpenVPN client to eth0'        >> /tmp/insert
echo '-A POSTROUTING -s 10.8.0.0/8 -o eth0 -j MASQUERADE' >> /tmp/insert
echo 'COMMIT'                                             >> /tmp/insert
echo '# END OPENVPN RULES'                                >> /tmp/insert
(sudo head -n 10 /etc/ufw/before.rules ; sudo cat /tmp/insert; sudo tail -n +10 /etc/ufw/before.rules) > /tmp/rules
sudo mv /tmp/rules /etc/ufw/before.rules
sudo ufw enable
sudo ufw disable

#zip ~/vpn/config.zip ~/vpn/config.ovpn
echo "Configuration file to use in OpenVPN clients to connect to $IP." | /usr/bin/mutt -a ~/vpn/config.ovpn -s "OpenVPN" -- pablo.aledo@gmail.com

# Transfer config.ovpn to client
# set DNS's
# config with openvpn --config salida.ovpn

