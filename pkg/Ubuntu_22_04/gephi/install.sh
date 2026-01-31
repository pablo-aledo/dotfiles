cd
wget https://github.com/gephi/gephi/releases/download/v0.10.1/gephi-0.10.1-linux-x64.tar.gz
tar -xvzf gephi-0.10.1-linux-x64.tar.gz
cd gephi-0.10.1
cd bin
sudo ln -s $PWD/gephi /usr/bin/gephi
