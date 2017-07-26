cd
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install -y --no-install-recommends libboost-all-dev
sudo apt-get install -y libatlas-base-dev 

pkg install libgflags-dev
pkg install libgoogle-glog-dev
pkg install liblmdb-dev
pkg install python-pip

git clone https://github.com/BVLC/caffe.git
cd caffe

cd python
for req in `cat requirements.txt`; do sudo pip install $req; done 
cd ..

mkdir build
cd build
cmake ..
/usr/bin/make -j`nproc`
make pycaffe -j`nproc`
sudo make install
make runtest -j`nproc`

echo 'export CAFFE_HOME=$HOME/caffe' >> ~/.paths
echo 'export PYTHONPATH=$HOME/caffe/pyton:$PYTHONPATH' >> ~/.paths
