cd
wget 'http://cilvr.cs.nyu.edu/lib/exe/fetch.php?media=overfeat:overfeat-v04-2.tgz' -O overfeat-v04-2.tgz

tar xvf overfeat-v04-2.tgz
cd overfeat

./download_weights.py

sudo apt-get install -y build-essential gcc g++ gfortran git libgfortran3
cd /tmp
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make NO_AFFINITY=1 USE_OPENMP=1
sudo make install

echo 'export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH' >> ~/.paths
