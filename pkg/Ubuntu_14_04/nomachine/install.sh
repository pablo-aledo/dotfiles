source ~/.dotfiles/source/.shell/links

cd /tmp/

for a in `seq 1 15`; do wget "https://www.nomachine.com/download/download&id=$a" ; done
file=`grep -Rin 'amd64' download\&id=* | cut -d ":" -f1 | head -n1`
dl=`links $file | grep amd64.deb`

wget $dl

sudo dpkg -i `basename $dl`
cd /tmp/

for a in `seq 1 15`; do wget "https://www.nomachine.com/download/download&id=$a" ; done
file=`grep -Rin 'amd64' download\&id=* | cut -d ":" -f1 | head -n1`
dl=`links $file | grep amd64.deb`

wget $dl

sudo dpkg -i `basename $dl`
