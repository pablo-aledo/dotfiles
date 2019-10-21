cd /tmp
wget https://storage.googleapis.com/harbor-releases/release-1.9.0/harbor-online-installer-v1.9.1.tgz
tar -xvzf harbor-online-installer-v1.9.1.tgz
cd harbor
ip=$(wget http://ipinfo.io/ip -qO -)
sed -i "s/hostname: reg.mydomain.com/hostname: $ip/g" harbor.yml
sudo ./install.sh
