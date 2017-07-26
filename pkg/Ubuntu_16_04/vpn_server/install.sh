FILES=`pwd`
sudo apt-get install -y openvpn easy-rsa

make-cadir ~/openvpn-ca
cd ~/openvpn-ca

cp $FILES/vars .

source vars

echo '. ./vars'                               >> /tmp/cmd
echo './clean-all'                            >> /tmp/cmd
echo './build-ca --batch'                     >> /tmp/cmd
echo './build-key-server --batch server'      >> /tmp/cmd
echo './build-dh'                             >> /tmp/cmd
echo 'openvpn --genkey --secret keys/ta.key'  >> /tmp/cmd
echo './build-key --batch client1'            >> /tmp/cmd
sudo bash /tmp/cmd

sudo cp keys/ca.crt keys/ca.key keys/server.crt keys/server.key keys/ta.key keys/dh2048.pem /etc/openvpn
gunzip -c /usr/share/doc/openvpn/examples/sample-config-files/server.conf.gz | sudo tee /etc/openvpn/server.conf
sudo cp $FILES/server.conf /etc/openvpn/server.conf
sudo cp $FILES/sysctl.conf /etc/sysctl.conf
sudo sysctl -p
sudo cp $FILES/before.rules /etc/ufw/before.rules
sudo cp $FILES/ufw /etc/default/ufw
sudo ufw allow 1194/udp
sudo ufw allow OpenSSH
sudo ufw disable
sudo ufw --force enable
sudo systemctl start openvpn@server
# sudo systemctl status openvpn@server
ip addr show tun0
sudo systemctl enable openvpn@server
mkdir -p ~/client-configs/files
chmod 700 ~/client-configs/files
cp /usr/share/doc/openvpn/examples/sample-config-files/client.conf ~/client-configs/base.conf
cp $FILES/base.conf ~/client-configs/base.conf

[ -e /usr/local/bin/noip2 ] && IP=pabloaledo.ddns.net
[ -e /usr/local/bin/noip2 ] || IP=`wget http://ipinfo.io/ip -qO -`
sed -i "s/%remote%/$IP/g" ~/client-configs/base.conf

cp $FILES/make_config.sh ~/client-configs/make_config.sh
chmod 700 ~/client-configs/make_config.sh
cd ~/client-configs
sudo ./make_config.sh client1

mkdir ~/vpn/ 
cp files/client1.ovpn ~/vpn/config.ovpn
