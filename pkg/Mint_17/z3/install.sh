cd /tmp

#git clone https://git01.codeplex.com/z3
wget 'http://download-codeplex.sec.s-msft.com/Download/SourceControlFileDownload.ashx?ProjectName=z3&changeSetId=cee7dd39444c9060186df79c2a2c7f8845de415b' -O z3.zip
mkdir z3; cd z3; unzip ../z3.zip

python scripts/mk_make.py
cd build
make
sudo make install
