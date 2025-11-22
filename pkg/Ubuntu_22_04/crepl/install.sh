cd /tmp
wget https://github.com/tsoding/crepl/archive/refs/heads/main.zip
unzip main.zip
cd crepl-main
sudo apt-get install -y libffi-dev
gcc crepl.c -lffi -o crepl
sudo cp crepl /usr/bin
