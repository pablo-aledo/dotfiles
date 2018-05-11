sudo apt-get update
sudo apt-get install -y build-essential vim-gnome python2.7 git libclang-dev

cd ~/ && git clone https://github.com/JBakamovic/yavide.git
cd yavide && ./install.sh
sudo rm -R ~/yavide
