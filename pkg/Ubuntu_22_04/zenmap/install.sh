sudo apt update
sudo apt install -y \
build-essential checkinstall \
zlib1g-dev libssl-dev \
libcurl4-openssl-dev \
python3 python3-pip \
git unzip

wget https://github.com/nmap/nmap/archive/refs/heads/master.zip -O nmap.zip
unzip nmap.zip
cd nmap-master

sed -i 's|\.\./nmap|nmap|g' zenmap/share/zenmap/config/zenmap.conf
sed -i 's|\.\./ndiff/ndiff||g' zenmap/share/zenmap/config/zenmap.conf

#python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --user --upgrade setuptools wheel pip
python3 -c "import setuptools; print(setuptools.__version__)"

./configure
make
sudo make install

cd zenmap; python3 -m pip install --user --no-build-isolation .

sudo -E env PYTHONPATH=$(pwd) python3 -m zenmapGUI.App
