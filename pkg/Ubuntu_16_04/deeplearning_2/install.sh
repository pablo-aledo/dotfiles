# Install cuda (finish installation)

sudo apt-get install -y build-essential
sudo apt-get install -y linux-image-extra-virtual
sudo apt-get install -y linux-headers-`uname -r`

wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run

sudo sh cuda_8.0.44_linux-run

echo 'export PATH=/usr/local/cuda-8.0/bin:$PATH'                         >> ~/.paths
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH' >> ~/.paths
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

cd ~/NVIDIA_CUDA-8.0_Samples/1_Utilities/deviceQuery
make
./deviceQuery

# Instal cudnn

cd

while true
do
	[ `ls | grep cudnn-8.0-linux-x64` ] || echo 'Downdload cudnn installer at https://developer.nvidia.com'
	[ `ls | grep cudnn` ] && break
	sleep 10
done
sleep 10

tar -xvzf cudnn-8.0-linux-x64-*.tgz
cd cuda
sudo cp lib64/* /usr/local/cuda/lib64/
sudo cp include/cudnn.h /usr/local/cuda/include/

# Install caffe

cd
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install -y --no-install-recommends libboost-all-dev
sudo apt-get install -y libatlas-base-dev 

pkg install libgflags-dev
pkg install libgoogle-glog-dev
pkg install liblmdb-dev
pkg install python-pip

sudo pip install --upgrade pip

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

echo 'export CAFFE_HOME=$HOME/caffe' >> ~/.paths
echo 'export PYTHONPATH=$HOME/caffe/python:$PYTHONPATH' >> ~/.paths

export CAFFE_HOME=$HOME/caffe
export PYTHONPATH=$HOME/caffe/python:$PYTHONPATH

# Install digits

cd 
sudo apt-get install -y --no-install-recommends git graphviz gunicorn python-dev python-flask python-flaskext.wtf python-gevent python-h5py python-numpy python-pil python-protobuf python-scipy

echo 'export DIGITS_HOME=~/digits' >> ~/.paths
#echo 'export DIGITS_ROOT=~/digits' >> ~/.paths
export DIGITS_HOME=~/digits
#export DIGITS_ROOT=~/digits

git clone https://github.com/NVIDIA/DIGITS.git $DIGITS_HOME

sudo pip install -r $DIGITS_HOME/requirements.txt

#sudo pip install -e $DIGITS_ROOT
#sudo apt-get install python-opencv

# Install tensorflow

sudo pip install tensorflow

# Install theano

sudo pip install Theano

# Install keras

sudo pip install keras

# Install jupyter

sudo apt-get install -y python-pip python-dev build-essential
sudo pip install jupyter markupsafe zmq singledispatch backports_abc certifi jsonschema path.py
 
# Install overfeat

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
export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH

# Install sklearn

sudo pip install -U scikit-learn

