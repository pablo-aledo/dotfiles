VERSION=2.1
sudo mkdir /usr/src/tmux; sudo chmod 777 /usr/src/tmux; cd /usr/src/tmux
sudo apt-get -y remove tmux
sudo apt-get -y install wget tar libevent-dev libncurses-dev
wget https://github.com/tmux/tmux/releases/download/$VERSION/tmux-$VERSION.tar.gz
tar xf tmux-$VERSION.tar.gz
rm -f tmux-$VERSION.tar.gz
cd tmux-$VERSION
./configure
make
sudo make install
cd -
sudo rm -rf /usr/local/src/tmux-*
sudo mv tmux-$VERSION /usr/local/src

## Logout and login to the shell again and run.
## tmux -V
