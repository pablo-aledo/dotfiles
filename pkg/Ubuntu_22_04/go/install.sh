ARCH=amd64
wget https://go.dev/dl/go1.20.4.linux-$ARCH.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.20.4.linux-$ARCH.tar.gz
sudo rm /usr/local/bin/go
sudo ln -s /usr/local/go/bin/go /usr/local/bin/go
rm -fr go1.20.4.linux-$ARCH.tar.gz
