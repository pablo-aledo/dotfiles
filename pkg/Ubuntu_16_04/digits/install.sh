cd 
sudo apt-get install -y --no-install-recommends git graphviz gunicorn python-dev python-flask python-flaskext.wtf python-gevent python-h5py python-numpy python-pil python-protobuf python-scipy

DIGITS_HOME=~/digits
echo 'export DIGITS_HOME=~/digits' >> ~/.paths
git clone https://github.com/NVIDIA/DIGITS.git $DIGITS_HOME

sudo pip install -r $DIGITS_HOME/requirements.txt

