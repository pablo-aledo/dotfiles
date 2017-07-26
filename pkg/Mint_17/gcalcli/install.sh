sudo apt-get install -y python-gflags python-dateutil python-httplib2 python-pip
sudo pip install --upgrade google-api-python-client
cd /
sudo git clone https://github.com/insanum/gcalcli.git
sudo ln -s /gcalcli/gcalcli /usr/local/bin/gcalcli
cd /gcalcli
sudo python setup.py install
