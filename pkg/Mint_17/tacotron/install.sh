pkg install python3
pkg install python3-pip
pip3 install tensorflow

cd
git clone https://github.com/keithito/tacotron
cd tacotron
pip3 install -r requirements.txt
curl http://data.keithito.com/data/speech/tacotron-20170720.tar.bz2 | tar xjC /tmp
python3 demo_server.py --checkpoint /tmp/tacotron-20170720/model.ckpt

# Point your browser at localhost:9000**
