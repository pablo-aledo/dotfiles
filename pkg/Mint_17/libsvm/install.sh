cd
wget http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+zip -O libsvm-3.21.zip
unzip libsvm-3.21.zip
cd libsvm-3.21
make
echo 'export PATH='$PWD':$PATH' >> ~/.paths
