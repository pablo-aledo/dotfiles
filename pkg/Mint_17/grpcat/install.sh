pkg install python3-pip
sudo pip3 install grpcio-tools
sudo pip3 install protobuf

cd
git clone https://github.com/GyroscopeHQ/grpcat.git
cd grpcat
sudo python3 setup.py install
sudo cp grpcat.py /usr/bin/
sudo chmod +x /usr/bin/grpcat.py
