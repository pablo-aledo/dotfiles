sudo apt-get install -y python-pip python-setuptools
cd
git clone https://github.com/deeplook/sparklines.git
cd sparklines
sudo pip install future
sudo python setup.py install
#sparklines 2 7 1 8 2 8 1 8
#sparklines -n 2 1 2 3 4 5.0 null 3 2 1

#sudo sh -c "curl https://raw.githubusercontent.com/holman/spark/master/spark -o /usr/local/bin/spark && chmod +x /usr/local/bin/spark"
